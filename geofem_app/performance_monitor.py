"""Performance monitoring and benchmark regression helpers."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fem2d_linear_solver import linear_factor_cache_info
from .fem2d_types import SolveResult2D, StageResult2D


PROFILE_CATEGORIES = (
    "assembly",
    "nonlinear_iteration",
    "linear_solve",
    "postprocess",
    "io_report",
    "cache_build",
    "cache_reuse",
    "coupled_assembly",
    "srm_trial",
)

_CATEGORY_TIME_KEYS: dict[str, tuple[str, ...]] = {
    "assembly": (
        "assembly_elapsed_seconds",
        "stiffness_assembly_elapsed_seconds",
        "tangent_assembly_elapsed_seconds",
        "internal_force_assembly_elapsed_seconds",
    ),
    "nonlinear_iteration": (
        "nonlinear_iteration_elapsed_seconds",
        "nonlinear_solve_elapsed_seconds",
        "newton_elapsed_seconds",
        "riks_elapsed_seconds",
    ),
    "linear_solve": ("linear_solve_elapsed_seconds", "solve_elapsed_seconds"),
    "postprocess": ("postprocess_elapsed_seconds", "result_postprocess_elapsed_seconds"),
    "io_report": ("io_report_elapsed_seconds", "report_elapsed_seconds", "output_elapsed_seconds"),
    "cache_build": ("cache_build_elapsed_seconds", "cache_setup_elapsed_seconds"),
    "cache_reuse": ("cache_reuse_elapsed_seconds",),
    "coupled_assembly": ("coupled_assembly_elapsed_seconds", "up_assembly_elapsed_seconds", "monolithic_assembly_elapsed_seconds"),
    "srm_trial": ("srm_trial_elapsed_seconds",),
}

_ITERATION_TIME_KEYS = (
    "elapsed_seconds",
    "iteration_elapsed_seconds",
    "assembly_elapsed_seconds",
    "tangent_internal_assembly_elapsed_seconds",
    "internal_force_elapsed_seconds",
    "internal_force_assembly_elapsed_seconds",
    "tangent_assembly_elapsed_seconds",
    "effective_stiffness_assembly_elapsed_seconds",
    "coupled_assembly_elapsed_seconds",
    "monolithic_assembly_elapsed_seconds",
    "reduced_matrix_elapsed_seconds",
    "augmented_bmat_elapsed_seconds",
    "lagrange_constraint_matrix_elapsed_seconds",
    "lagrange_bmat_elapsed_seconds",
    "lagrange_linear_solve_elapsed_seconds",
    "linear_solve_elapsed_seconds",
    "line_search_elapsed_seconds",
    "postprocess_elapsed_seconds",
)

_ITERATION_RESIDUAL_KEYS = (
    "residual_norm",
    "force_residual_norm",
    "pressure_residual_norm",
    "constraint_norm",
    "mass_balance_residual_sum",
)


def build_performance_summary(
    result: SolveResult2D,
    *,
    elapsed_seconds: float | None = None,
    case_name: str | None = None,
) -> dict[str, Any]:
    stages = [_stage_performance_row(stage, len(result.mesh.node_ids)) for stage in result.stages]
    profile = _summary_performance_profile(stages)
    stage_elapsed = sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in stages)
    total_elapsed = elapsed_seconds if elapsed_seconds is not None else stage_elapsed
    run_io_profile = _result_run_io_profile(result)
    run_io_report_elapsed = _float(run_io_profile.get("run_io_report_elapsed_seconds"), 0.0)
    stage_io_report_elapsed = sum(float(row.get("io_report_elapsed_seconds", 0.0) or 0.0) for row in stages)
    total_io_report_elapsed = stage_io_report_elapsed + run_io_report_elapsed
    if total_io_report_elapsed > 0.0:
        categories = profile.get("category_totals", {}) if isinstance(profile.get("category_totals", {}), dict) else {}
        categories["io_report"] = total_io_report_elapsed
        profile["category_totals"] = categories
        dominant_category, dominant_elapsed = _dominant_category(categories)
        profile["dominant_category"] = dominant_category
        profile["dominant_category_elapsed_seconds"] = dominant_elapsed
    solve_elapsed_excluding_io = max(float(total_elapsed or 0.0) - total_io_report_elapsed, 0.0)
    total_iterations = sum(int(row.get("solver_iterations", 0) or 0) for row in stages)
    max_matrix_nnz = max((int(row.get("matrix_nnz", 0) or 0) for row in stages), default=0)
    estimated_bytes = sum(int(row.get("estimated_memory_bytes", 0) or 0) for row in stages) + _result_array_bytes(result)
    assembly_elapsed = sum(float(row.get("assembly_elapsed_seconds", 0.0) or 0.0) for row in stages)
    nonlinear_iteration_elapsed = sum(float(row.get("nonlinear_iteration_elapsed_seconds", 0.0) or 0.0) for row in stages)
    linear_solve_elapsed = sum(float(row.get("linear_solve_elapsed_seconds", 0.0) or 0.0) for row in stages)
    postprocess_elapsed = sum(float(row.get("postprocess_elapsed_seconds", 0.0) or 0.0) for row in stages)
    io_report_elapsed = total_io_report_elapsed
    cache_build_elapsed = sum(float(row.get("cache_build_elapsed_seconds", 0.0) or 0.0) for row in stages)
    cache_reuse_elapsed = sum(float(row.get("cache_reuse_elapsed_seconds", 0.0) or 0.0) for row in stages)
    coupled_assembly_elapsed = sum(float(row.get("coupled_assembly_elapsed_seconds", 0.0) or 0.0) for row in stages)
    srm_trial_elapsed = sum(float(row.get("srm_trial_elapsed_seconds", 0.0) or 0.0) for row in stages)
    return {
        "schema": "geofem.performance_summary.v1",
        "case": case_name or "",
        "elapsed_seconds": float(total_elapsed or 0.0),
        "stage_elapsed_seconds": float(stage_elapsed or 0.0),
        "cold_run_elapsed_seconds": "" if elapsed_seconds is None else float(elapsed_seconds or 0.0),
        "solve_elapsed_seconds_excluding_io": float(solve_elapsed_excluding_io),
        "cold_solve_elapsed_seconds_excluding_io": "" if elapsed_seconds is None else float(solve_elapsed_excluding_io),
        "stage_io_report_elapsed_seconds": float(stage_io_report_elapsed),
        "run_io_report_elapsed_seconds": float(run_io_report_elapsed),
        "cold_io_report_elapsed_seconds": float(total_io_report_elapsed),
        "stage_count": len(stages),
        "node_count": len(result.mesh.node_ids),
        "element_count": len(result.mesh.elements),
        "total_solver_iterations": total_iterations,
        "max_matrix_nnz": max_matrix_nnz,
        "estimated_memory_bytes": estimated_bytes,
        "assembly_elapsed_seconds": assembly_elapsed,
        "nonlinear_iteration_elapsed_seconds": nonlinear_iteration_elapsed,
        "linear_solve_elapsed_seconds": linear_solve_elapsed,
        "postprocess_elapsed_seconds": postprocess_elapsed,
        "io_report_elapsed_seconds": io_report_elapsed,
        "cache_build_elapsed_seconds": cache_build_elapsed,
        "cache_reuse_elapsed_seconds": cache_reuse_elapsed,
        "coupled_assembly_elapsed_seconds": coupled_assembly_elapsed,
        "srm_trial_elapsed_seconds": srm_trial_elapsed,
        "dominant_category": profile["dominant_category"],
        "dominant_category_elapsed_seconds": profile["dominant_category_elapsed_seconds"],
        "dominant_stage": profile["dominant_stage"],
        "cache_entry_count": profile["cache"]["entry_count"],
        "cache_reuse_count": profile["cache"]["reuse_count"],
        "cache_build_count": profile["cache"]["build_count"],
        "cache_hit_count": profile["cache"]["hit_count"],
        "cache_miss_count": profile["cache"]["miss_count"],
        "cache_unused_reasons": ";".join(profile["cache"]["unused_reasons"]),
        "iteration_profile_count": profile["iterations"]["count"],
        "slowest_iteration_path": profile["iterations"]["slowest"].get("path", ""),
        "slowest_iteration_elapsed_seconds": profile["iterations"]["slowest"].get("elapsed_seconds", ""),
        "linear_factor_cache": linear_factor_cache_info(),
        "run_io_profile": run_io_profile,
        "profile": profile,
        "stages": stages,
    }


def write_performance_summary(
    result: SolveResult2D,
    output_dir: str | Path | None = None,
    *,
    elapsed_seconds: float | None = None,
    case_name: str | None = None,
) -> dict[str, str]:
    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = build_performance_summary(result, elapsed_seconds=elapsed_seconds, case_name=case_name)
    json_path = out / "performance_summary.json"
    csv_path = out / "performance_summary.csv"
    html_path = out / "performance_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fields = [
        "stage",
        "method",
        "linear_method",
        "elapsed_seconds",
        "dof_count",
        "matrix_size",
        "matrix_nnz",
        "constrained_dofs",
        "free_dofs",
        "solver_iterations",
        "residual_norm",
        "estimated_memory_bytes",
        "assembly_elapsed_seconds",
        "nonlinear_iteration_elapsed_seconds",
        "linear_solve_elapsed_seconds",
        "postprocess_elapsed_seconds",
        "io_report_elapsed_seconds",
        "cache_build_elapsed_seconds",
        "cache_reuse_elapsed_seconds",
        "geometry_mode",
        "element_type",
        "integration",
        "material_model",
        "batched_elements",
        "fallback_count",
        "fallback_reasons",
        "coupled_assembly_elapsed_seconds",
        "srm_trial_elapsed_seconds",
        "srm_trial_solver_elapsed_seconds",
        "srm_trial_overhead_elapsed_seconds",
        "srm_slowest_trial_elapsed_seconds",
        "dominant_category",
        "dominant_category_elapsed_seconds",
        "cache_entry_count",
        "cache_reuse_count",
        "cache_build_count",
        "cache_hit_count",
        "cache_miss_count",
        "cache_unused_reasons",
        "iteration_profile_count",
        "slowest_iteration_path",
        "slowest_iteration_elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary["stages"]:
            writer.writerow({field: row.get(field, "") for field in fields})
    html_path.write_text(_performance_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def benchmark_case_performance(
    case_name: str,
    category: str,
    result: SolveResult2D,
    elapsed_seconds: float,
    *,
    passed: bool,
) -> dict[str, Any]:
    summary = build_performance_summary(result, elapsed_seconds=elapsed_seconds, case_name=case_name)
    profile = summary.get("profile", {}) if isinstance(summary.get("profile", {}), Mapping) else {}
    cache = profile.get("cache", {}) if isinstance(profile.get("cache", {}), Mapping) else {}
    iterations = profile.get("iterations", {}) if isinstance(profile.get("iterations", {}), Mapping) else {}
    slowest = iterations.get("slowest", {}) if isinstance(iterations.get("slowest", {}), Mapping) else {}
    return {
        "case": case_name,
        "category": category,
        "passed": bool(passed),
        "elapsed_seconds": summary["elapsed_seconds"],
        "stage_elapsed_seconds": summary["stage_elapsed_seconds"],
        "cold_run_elapsed_seconds": elapsed_seconds,
        "solve_elapsed_seconds_excluding_io": summary["solve_elapsed_seconds_excluding_io"],
        "cold_solve_elapsed_seconds_excluding_io": summary["cold_solve_elapsed_seconds_excluding_io"],
        "cold_io_report_elapsed_seconds": summary["cold_io_report_elapsed_seconds"],
        "stage_io_report_elapsed_seconds": summary["stage_io_report_elapsed_seconds"],
        "run_io_report_elapsed_seconds": summary["run_io_report_elapsed_seconds"],
        "stage_count": summary["stage_count"],
        "node_count": summary["node_count"],
        "element_count": summary["element_count"],
        "total_solver_iterations": summary["total_solver_iterations"],
        "max_matrix_nnz": summary["max_matrix_nnz"],
        "estimated_memory_bytes": summary["estimated_memory_bytes"],
        "fallback_count": sum(int(row.get("fallback_count", 0) or 0) for row in summary["stages"]),
        "batched_elements": sum(int(row.get("batched_elements", 0) or 0) for row in summary["stages"]),
        "assembly_elapsed_seconds": summary["assembly_elapsed_seconds"],
        "nonlinear_iteration_elapsed_seconds": summary["nonlinear_iteration_elapsed_seconds"],
        "linear_solve_elapsed_seconds": summary["linear_solve_elapsed_seconds"],
        "postprocess_elapsed_seconds": summary["postprocess_elapsed_seconds"],
        "io_report_elapsed_seconds": summary["io_report_elapsed_seconds"],
        "cache_build_elapsed_seconds": summary["cache_build_elapsed_seconds"],
        "cache_reuse_elapsed_seconds": summary["cache_reuse_elapsed_seconds"],
        "coupled_assembly_elapsed_seconds": summary["coupled_assembly_elapsed_seconds"],
        "srm_trial_elapsed_seconds": summary["srm_trial_elapsed_seconds"],
        "dominant_category": summary["dominant_category"],
        "dominant_category_elapsed_seconds": summary["dominant_category_elapsed_seconds"],
        "dominant_stage": summary["dominant_stage"],
        "cache_entry_count": int(cache.get("entry_count", 0) or 0),
        "cache_reuse_count": int(cache.get("reuse_count", 0) or 0),
        "cache_build_count": int(cache.get("build_count", 0) or 0),
        "cache_hit_count": int(cache.get("hit_count", 0) or 0),
        "cache_miss_count": int(cache.get("miss_count", 0) or 0),
        "cache_unused_reasons": ";".join(str(reason) for reason in cache.get("unused_reasons", []) if str(reason)) if isinstance(cache.get("unused_reasons", []), list) else "",
        "iteration_profile_count": int(iterations.get("count", 0) or 0),
        "slowest_iteration_path": slowest.get("path", ""),
        "slowest_iteration_elapsed_seconds": slowest.get("elapsed_seconds", ""),
    }


def compare_performance_cases(
    current_cases: list[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
    *,
    max_slowdown: float = 1.25,
    max_structure_growth: float | None = None,
    max_memory_growth: float | None = None,
    max_fallback_increase: int = 0,
) -> list[dict[str, Any]]:
    if not baseline:
        return []
    baseline_rows = baseline.get("cases", baseline.get("performance_cases", []))
    if not isinstance(baseline_rows, list):
        return []
    base_by_name = {str(row.get("case", row.get("name", ""))): row for row in baseline_rows if isinstance(row, Mapping)}
    regressions: list[dict[str, Any]] = []
    structure_limit = float(max_slowdown if max_structure_growth is None else max_structure_growth)
    memory_limit = float(max_slowdown if max_memory_growth is None else max_memory_growth)
    for row in current_cases:
        name = str(row.get("case", row.get("name", "")))
        base = base_by_name.get(name)
        if not base:
            continue
        current_time = _float(row.get("elapsed_seconds"), 0.0)
        baseline_time = _float(base.get("elapsed_seconds"), 0.0)
        if baseline_time <= 0.0:
            continue
        slowdown = current_time / baseline_time
        if slowdown > max_slowdown:
            regressions.append(
                {
                    "case": name,
                    "metric": "elapsed_seconds",
                    "current": current_time,
                    "baseline": baseline_time,
                    "slowdown": slowdown,
                    "max_slowdown": max_slowdown,
                }
            )
        current_iter = _float(row.get("total_solver_iterations"), 0.0)
        baseline_iter = _float(base.get("total_solver_iterations"), 0.0)
        if baseline_iter > 0.0 and current_iter / baseline_iter > max_slowdown:
            regressions.append(
                {
                    "case": name,
                    "metric": "total_solver_iterations",
                    "current": current_iter,
                    "baseline": baseline_iter,
                    "slowdown": current_iter / baseline_iter,
                    "max_slowdown": max_slowdown,
                }
            )
        for metric, limit in (("max_matrix_nnz", structure_limit), ("estimated_memory_bytes", memory_limit)):
            current_value = _float(row.get(metric), 0.0)
            baseline_value = _float(base.get(metric), 0.0)
            if baseline_value <= 0.0:
                continue
            growth = current_value / baseline_value
            if growth > limit:
                regressions.append(
                    {
                        "case": name,
                        "metric": metric,
                        "current": current_value,
                        "baseline": baseline_value,
                        "growth": growth,
                        "max_growth": limit,
                    }
                )
        current_fallback = int(_float(row.get("fallback_count"), 0.0))
        baseline_fallback = int(_float(base.get("fallback_count"), 0.0))
        if current_fallback > baseline_fallback + max(int(max_fallback_increase), 0):
            regressions.append(
                {
                    "case": name,
                    "metric": "fallback_count",
                    "current": current_fallback,
                    "baseline": baseline_fallback,
                    "increase": current_fallback - baseline_fallback,
                    "max_increase": max(int(max_fallback_increase), 0),
                }
            )
        current_builder = int(_float(row.get("sparse_builder_to_csr_count"), 0.0))
        baseline_builder = int(_float(base.get("sparse_builder_to_csr_count"), 0.0))
        if current_builder > baseline_builder:
            regressions.append(
                {
                    "case": name,
                    "metric": "sparse_builder_to_csr_count",
                    "current": current_builder,
                    "baseline": baseline_builder,
                    "increase": current_builder - baseline_builder,
                    "max_increase": 0,
                }
            )
    return regressions


def write_benchmark_performance_reports(
    cases: list[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    baseline: Mapping[str, Any] | None = None,
    max_slowdown: float = 1.25,
    max_structure_growth: float | None = None,
    max_memory_growth: float | None = None,
    max_fallback_increase: int = 0,
    update_baseline: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    regressions = compare_performance_cases(
        cases,
        baseline,
        max_slowdown=max_slowdown,
        max_structure_growth=max_structure_growth,
        max_memory_growth=max_memory_growth,
        max_fallback_increase=max_fallback_increase,
    )
    report = {
        "schema": "geofem.benchmark_performance.v1",
        "case_count": len(cases),
        "regression_count": len(regressions),
        "max_slowdown": max_slowdown,
        "max_structure_growth": max_slowdown if max_structure_growth is None else max_structure_growth,
        "max_memory_growth": max_slowdown if max_memory_growth is None else max_memory_growth,
        "max_fallback_increase": max_fallback_increase,
        "passed": len(regressions) == 0,
        "cases": [dict(row) for row in cases],
        "regressions": regressions,
    }
    json_path = out / "standard_benchmark_performance.json"
    csv_path = out / "standard_benchmark_performance.csv"
    html_path = out / "standard_benchmark_performance.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fields = [
        "case",
        "category",
        "passed",
        "elapsed_seconds",
        "stage_elapsed_seconds",
        "cold_run_elapsed_seconds",
        "solve_elapsed_seconds_excluding_io",
        "cold_solve_elapsed_seconds_excluding_io",
        "cold_io_report_elapsed_seconds",
        "stage_io_report_elapsed_seconds",
        "run_io_report_elapsed_seconds",
        "stage_count",
        "node_count",
        "element_count",
        "total_solver_iterations",
        "max_matrix_nnz",
        "estimated_memory_bytes",
        "fallback_count",
        "batched_elements",
        "assembly_elapsed_seconds",
        "nonlinear_iteration_elapsed_seconds",
        "linear_solve_elapsed_seconds",
        "postprocess_elapsed_seconds",
        "io_report_elapsed_seconds",
        "cache_build_elapsed_seconds",
        "cache_reuse_elapsed_seconds",
        "coupled_assembly_elapsed_seconds",
        "srm_trial_elapsed_seconds",
        "dominant_category",
        "dominant_category_elapsed_seconds",
        "dominant_stage",
        "cache_entry_count",
        "cache_reuse_count",
        "cache_build_count",
        "cache_hit_count",
        "cache_miss_count",
        "cache_unused_reasons",
        "iteration_profile_count",
        "slowest_iteration_path",
        "slowest_iteration_elapsed_seconds",
        "sparse_pattern_build_count",
        "sparse_pattern_assemble_count",
        "sparse_duplicate_scatter_count",
        "sparse_builder_to_csr_count",
        "sparse_builder_block_count",
        "sparse_builder_value_count",
        "sparse_builder_elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in cases:
            writer.writerow({field: row.get(field, "") for field in fields})
    html_path.write_text(_benchmark_performance_html(report), encoding="utf-8")
    paths = {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}
    if update_baseline:
        baseline_path = out / "standard_benchmark_performance_baseline.json"
        baseline_path.write_text(json.dumps({"schema": "geofem.benchmark_performance_baseline.v1", "cases": report["cases"]}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        paths["baseline_json"] = str(baseline_path)
    return {**report, "paths": paths}


def _stage_performance_row(stage: StageResult2D, node_count: int) -> dict[str, Any]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    performance = solver.get("performance", {}) if isinstance(solver.get("performance", {}), Mapping) else {}
    matrix = solver.get("matrix", {}) if isinstance(solver.get("matrix", {}), Mapping) else {}
    dof_count = int(matrix.get("size", stage.displacements.size) or stage.displacements.size)
    nnz = int(matrix.get("nnz", 0) or 0)
    profile = _stage_performance_profile(stage.name, solver, performance)
    cache_profile = profile["cache"]
    iteration_profile = profile["iterations"]
    slowest_iteration = iteration_profile["slowest"] if isinstance(iteration_profile.get("slowest", {}), Mapping) else {}
    row = {
        "stage": stage.name,
        "method": solver.get("method", ""),
        "linear_method": solver.get("linear_method", solver.get("method", "")),
        "elapsed_seconds": float(performance.get("elapsed_seconds", 0.0) or 0.0),
        "dof_count": dof_count,
        "nodal_dof_count": node_count * 2,
        "matrix_size": matrix.get("size", dof_count),
        "matrix_nnz": nnz,
        "constrained_dofs": matrix.get("constrained_dofs", len(stage.constrained_dofs)),
        "free_dofs": matrix.get("free_dofs", max(dof_count - len(stage.constrained_dofs), 0)),
        "solver_iterations": int(solver.get("iterations", 0) or 0),
        "residual_norm": solver.get("residual_norm", ""),
        "estimated_memory_bytes": int(matrix.get("estimated_sparse_bytes", _estimated_sparse_bytes(dof_count, nnz)) or 0) + int(stage.displacements.nbytes + stage.reactions.nbytes),
        "fallback_count": int(solver.get("fallback_count", 0) or 0),
        "fallback_reasons": ";".join(str(reason) for reason in solver.get("fallback_reasons", []) if str(reason)) if isinstance(solver.get("fallback_reasons", []), list) else "",
        "batched_elements": int(solver.get("batched_elements", 0) or 0),
        "geometry_mode": solver.get("geometry_mode", ""),
        "element_type": solver.get("element_type", ""),
        "integration": solver.get("integration", ""),
        "material_model": solver.get("material_model", ""),
        "assembly_elapsed_seconds": float(performance.get("assembly_elapsed_seconds", 0.0) or 0.0),
        "nonlinear_iteration_elapsed_seconds": float(profile["category_totals"].get("nonlinear_iteration", 0.0) or 0.0),
        "linear_solve_elapsed_seconds": float(performance.get("linear_solve_elapsed_seconds", 0.0) or 0.0),
        "postprocess_elapsed_seconds": float(performance.get("postprocess_elapsed_seconds", 0.0) or 0.0),
        "io_report_elapsed_seconds": float(profile["category_totals"].get("io_report", 0.0) or 0.0),
        "cache_build_elapsed_seconds": float(profile["category_totals"].get("cache_build", 0.0) or 0.0),
        "cache_reuse_elapsed_seconds": float(profile["category_totals"].get("cache_reuse", 0.0) or 0.0),
        "coupled_assembly_elapsed_seconds": float(performance.get("coupled_assembly_elapsed_seconds", performance.get("up_assembly_elapsed_seconds", 0.0)) or 0.0),
        "srm_trial_elapsed_seconds": float(performance.get("srm_trial_elapsed_seconds", 0.0) or 0.0),
        "srm_trial_solver_elapsed_seconds": float(performance.get("srm_trial_solver_elapsed_seconds", 0.0) or 0.0),
        "srm_trial_overhead_elapsed_seconds": float(performance.get("srm_trial_overhead_elapsed_seconds", 0.0) or 0.0),
        "srm_slowest_trial_elapsed_seconds": float(performance.get("srm_slowest_trial_elapsed_seconds", 0.0) or 0.0),
        "dominant_category": profile["dominant_category"],
        "dominant_category_elapsed_seconds": profile["dominant_category_elapsed_seconds"],
        "cache_entry_count": int(cache_profile.get("entry_count", 0) or 0),
        "cache_reuse_count": int(cache_profile.get("reuse_count", 0) or 0),
        "cache_build_count": int(cache_profile.get("build_count", 0) or 0),
        "cache_hit_count": int(cache_profile.get("hit_count", 0) or 0),
        "cache_miss_count": int(cache_profile.get("miss_count", 0) or 0),
        "cache_unused_reasons": ";".join(str(reason) for reason in cache_profile.get("unused_reasons", []) if str(reason)) if isinstance(cache_profile.get("unused_reasons", []), list) else "",
        "iteration_profile_count": int(iteration_profile.get("count", 0) or 0),
        "slowest_iteration_path": slowest_iteration.get("path", ""),
        "slowest_iteration_elapsed_seconds": slowest_iteration.get("elapsed_seconds", ""),
        "profile": profile,
    }
    return row


def _stage_performance_profile(stage_name: str, solver: Mapping[str, Any], performance: Mapping[str, Any]) -> dict[str, Any]:
    category_totals = {category: _category_elapsed(performance, category) for category in PROFILE_CATEGORIES}
    dominant_category, dominant_elapsed = _dominant_category(category_totals)
    cache_entries = _collect_cache_profiles(solver)
    iteration_entries = _collect_iteration_profiles(solver)
    return {
        "schema": "geofem.performance_profile.stage.v1",
        "stage": stage_name,
        "category_totals": category_totals,
        "dominant_category": dominant_category,
        "dominant_category_elapsed_seconds": dominant_elapsed,
        "cache": _cache_profile_summary(cache_entries),
        "iterations": {
            "count": len(iteration_entries),
            "slowest": _slowest_iteration(iteration_entries),
            "entries": iteration_entries,
        },
    }


def _summary_performance_profile(stages: list[Mapping[str, Any]]) -> dict[str, Any]:
    category_totals = {category: 0.0 for category in PROFILE_CATEGORIES}
    cache_entries: list[dict[str, Any]] = []
    iteration_entries: list[dict[str, Any]] = []
    stage_profiles: list[Mapping[str, Any]] = []
    for row in stages:
        profile = row.get("profile", {}) if isinstance(row.get("profile", {}), Mapping) else {}
        stage_profiles.append(profile)
        categories = profile.get("category_totals", {}) if isinstance(profile.get("category_totals", {}), Mapping) else {}
        for category in PROFILE_CATEGORIES:
            category_totals[category] += _float(categories.get(category), 0.0)
        cache = profile.get("cache", {}) if isinstance(profile.get("cache", {}), Mapping) else {}
        entries = cache.get("entries", []) if isinstance(cache.get("entries", []), list) else []
        cache_entries.extend(entry for entry in entries if isinstance(entry, dict))
        iterations = profile.get("iterations", {}) if isinstance(profile.get("iterations", {}), Mapping) else {}
        entries = iterations.get("entries", []) if isinstance(iterations.get("entries", []), list) else []
        iteration_entries.extend(entry for entry in entries if isinstance(entry, dict))
    dominant_category, dominant_elapsed = _dominant_category(category_totals)
    slowest_stage = max(stages, key=lambda row: _float(row.get("elapsed_seconds"), 0.0), default={})
    return {
        "schema": "geofem.performance_profile.v1",
        "category_totals": category_totals,
        "dominant_category": dominant_category,
        "dominant_category_elapsed_seconds": dominant_elapsed,
        "dominant_stage": str(slowest_stage.get("stage", "")) if isinstance(slowest_stage, Mapping) else "",
        "slowest_stage": {
            "stage": str(slowest_stage.get("stage", "")) if isinstance(slowest_stage, Mapping) else "",
            "elapsed_seconds": _float(slowest_stage.get("elapsed_seconds"), 0.0) if isinstance(slowest_stage, Mapping) else 0.0,
        },
        "cache": _cache_profile_summary(cache_entries),
        "iterations": {
            "count": len(iteration_entries),
            "slowest": _slowest_iteration(iteration_entries),
        },
        "stages": stage_profiles,
    }


def _category_elapsed(performance: Mapping[str, Any], category: str) -> float:
    for key in _CATEGORY_TIME_KEYS.get(category, ()):
        if key in performance:
            return _float(performance.get(key), 0.0)
    return 0.0


def _result_run_io_profile(result: SolveResult2D) -> dict[str, Any]:
    raw = getattr(result, "run_io_profile", {})
    if not isinstance(raw, Mapping):
        return {"schema": "geofem.run_io_profile.v1", "run_io_report_elapsed_seconds": 0.0, "entries": {}}
    entries = raw.get("entries", {}) if isinstance(raw.get("entries", {}), Mapping) else {}
    run_elapsed = _float(raw.get("run_io_report_elapsed_seconds"), sum(_float(value, 0.0) for value in entries.values()))
    return {
        "schema": str(raw.get("schema", "geofem.run_io_profile.v1")),
        "stage_io_report_elapsed_seconds": _float(raw.get("stage_io_report_elapsed_seconds"), 0.0),
        "run_io_report_elapsed_seconds": run_elapsed,
        "io_report_elapsed_seconds": _float(raw.get("io_report_elapsed_seconds"), run_elapsed),
        "entries": {str(key): _float(value, 0.0) for key, value in entries.items()},
    }


def _dominant_category(category_totals: Mapping[str, Any]) -> tuple[str, float]:
    if not category_totals:
        return "", 0.0
    name, value = max(((str(key), _float(value, 0.0)) for key, value in category_totals.items()), key=lambda item: item[1])
    return (name, value) if value > 0.0 else ("", 0.0)


def _collect_cache_profiles(
    value: Any,
    *,
    path: str = "solver_info",
    limit: int = 200,
    _seen: set[int] | None = None,
) -> list[dict[str, Any]]:
    seen = _seen if _seen is not None else set()
    if len(seen) > 2000:
        return []
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            return []
        seen.add(object_id)
        entries: list[dict[str, Any]] = []
        if _looks_like_cache_mapping(path, value):
            entries.append(_normalize_cache_entry(path, value))
            if len(entries) >= limit:
                return entries[:limit]
        for key, child in value.items():
            if len(entries) >= limit:
                break
            if isinstance(child, (Mapping, list, tuple)):
                entries.extend(_collect_cache_profiles(child, path=f"{path}.{key}", limit=limit - len(entries), _seen=seen))
        return entries[:limit]
    if isinstance(value, (list, tuple)):
        entries: list[dict[str, Any]] = []
        for index, child in enumerate(value):
            if len(entries) >= limit:
                break
            if isinstance(child, (Mapping, list, tuple)):
                entries.extend(_collect_cache_profiles(child, path=f"{path}[{index}]", limit=limit - len(entries), _seen=seen))
        return entries[:limit]
    return []


def _looks_like_cache_mapping(path: str, value: Mapping[str, Any]) -> bool:
    lower_path = path.lower()
    if "cache" not in lower_path and "cache_kind" not in value:
        return False
    keys = {str(key).lower() for key in value.keys()}
    if "enabled" in keys or "cache_kind" in keys or "state" in keys:
        return True
    cache_markers = (
        "hits",
        "misses",
        "builds",
        "reuses",
        "reused",
        "built",
        "cached",
        "direct_fill",
    )
    return any(marker in key for key in keys for marker in cache_markers)


def _normalize_cache_entry(path: str, value: Mapping[str, Any]) -> dict[str, Any]:
    state = str(value.get("state", ""))
    enabled = bool(value.get("enabled", True if value.get("cache_kind") else False))
    hit_count = _cache_count(value, suffixes=("_hits",), exact=("hits", "hit_count", "cache_hits"))
    miss_count = _cache_count(value, suffixes=("_misses",), exact=("misses", "miss_count", "cache_misses"))
    build_count = _cache_count(value, suffixes=("_builds",), bool_suffixes=("_cached",), exact=("builds", "build_count", "built"))
    reuse_count = hit_count + _cache_count(value, suffixes=("_reuses",), bool_suffixes=("_reused",), exact=("reuses", "reuse_count", "reused"))
    if state.lower() == "hit":
        hit_count += 1
        reuse_count += 1
    elif state.lower() == "miss":
        miss_count += 1
    return {
        "path": path,
        "cache_kind": str(value.get("cache_kind", path.rsplit(".", 1)[-1])),
        "enabled": enabled,
        "state": state,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "build_count": build_count,
        "reuse_count": reuse_count,
        "direct_fill": _direct_fill_enabled(value),
        "unused_reasons": _cache_unused_reasons(path, value, enabled),
    }


def _cache_count(
    value: Mapping[str, Any],
    *,
    exact: tuple[str, ...] = (),
    suffixes: tuple[str, ...] = (),
    bool_suffixes: tuple[str, ...] = (),
) -> int:
    total = 0
    exact_set = {key.lower() for key in exact}
    for key, item in value.items():
        lower = str(key).lower()
        if lower in exact_set or any(lower.endswith(suffix) for suffix in suffixes):
            total += _int_count(item)
        elif isinstance(item, bool) and any(lower.endswith(suffix) for suffix in bool_suffixes):
            total += int(item)
    return total


def _int_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _direct_fill_enabled(value: Mapping[str, Any]) -> bool:
    direct = value.get("direct_fill", False)
    if isinstance(direct, Mapping):
        return bool(direct.get("enabled", False))
    return bool(direct)


def _cache_unused_reasons(path: str, value: Mapping[str, Any], enabled: bool) -> list[str]:
    reasons: list[str] = []
    if not enabled:
        reasons.append(f"{path}=disabled")
    markers = ("not", "disable", "fallback", "rebuild", "invalid", "missing", "changed", "uncached")
    for key, item in value.items():
        if not str(key).lower().endswith("reason") or not isinstance(item, str) or not item:
            continue
        lower = item.lower()
        if any(marker in lower for marker in markers):
            reasons.append(f"{path}.{key}={item}")
    return reasons


def _cache_profile_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for entry in entries:
        reasons.extend(str(reason) for reason in entry.get("unused_reasons", []) if str(reason))
    return {
        "entry_count": len(entries),
        "reuse_count": sum(int(entry.get("reuse_count", 0) or 0) for entry in entries),
        "build_count": sum(int(entry.get("build_count", 0) or 0) for entry in entries),
        "hit_count": sum(int(entry.get("hit_count", 0) or 0) for entry in entries),
        "miss_count": sum(int(entry.get("miss_count", 0) or 0) for entry in entries),
        "direct_fill_count": sum(1 for entry in entries if bool(entry.get("direct_fill", False))),
        "unused_reasons": sorted(set(reasons)),
        "entries": entries,
    }


def _collect_iteration_profiles(
    value: Any,
    *,
    path: str = "solver_info",
    limit: int = 500,
    _seen: set[int] | None = None,
) -> list[dict[str, Any]]:
    seen = _seen if _seen is not None else set()
    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            return []
        seen.add(object_id)
        entries: list[dict[str, Any]] = []
        for key, child in value.items():
            if len(entries) >= limit:
                break
            child_path = f"{path}.{key}"
            if isinstance(child, (list, tuple)) and _list_has_iteration_rows(child):
                for index, item in enumerate(child):
                    if len(entries) >= limit:
                        break
                    if isinstance(item, Mapping) and _looks_like_iteration_row(item):
                        entries.append(_normalize_iteration_entry(f"{child_path}[{index}]", item))
            elif isinstance(child, (Mapping, list, tuple)):
                entries.extend(_collect_iteration_profiles(child, path=child_path, limit=limit - len(entries), _seen=seen))
        return entries[:limit]
    if isinstance(value, (list, tuple)):
        entries: list[dict[str, Any]] = []
        for index, child in enumerate(value):
            if len(entries) >= limit:
                break
            if isinstance(child, Mapping) and _looks_like_iteration_row(child):
                entries.append(_normalize_iteration_entry(f"{path}[{index}]", child))
            elif isinstance(child, (Mapping, list, tuple)):
                entries.extend(_collect_iteration_profiles(child, path=f"{path}[{index}]", limit=limit - len(entries), _seen=seen))
        return entries[:limit]
    return []


def _list_has_iteration_rows(value: list[Any] | tuple[Any, ...]) -> bool:
    for item in value[:5]:
        if isinstance(item, Mapping) and _looks_like_iteration_row(item):
            return True
    return False


def _looks_like_iteration_row(value: Mapping[str, Any]) -> bool:
    keys = {str(key).lower() for key in value.keys()}
    has_index = bool(keys & {"iteration", "step", "increment", "newton_iteration", "outer_iteration"})
    has_metric = bool(keys & set(_ITERATION_TIME_KEYS)) or bool(keys & set(_ITERATION_RESIDUAL_KEYS)) or "converged" in keys
    return has_index and has_metric


def _normalize_iteration_entry(path: str, value: Mapping[str, Any]) -> dict[str, Any]:
    elapsed = _first_numeric(value, *_ITERATION_TIME_KEYS)
    residual = _first_numeric(value, *_ITERATION_RESIDUAL_KEYS)
    return {
        "path": path,
        "iteration": value.get("iteration", value.get("step", value.get("increment", value.get("newton_iteration", value.get("outer_iteration", ""))))),
        "elapsed_seconds": "" if elapsed is None else elapsed,
        "assembly_elapsed_seconds": _optional_number(value.get("assembly_elapsed_seconds")),
        "tangent_internal_assembly_elapsed_seconds": _optional_number(value.get("tangent_internal_assembly_elapsed_seconds")),
        "tangent_assembly_elapsed_seconds": _optional_number(value.get("tangent_assembly_elapsed_seconds")),
        "internal_force_assembly_elapsed_seconds": _optional_number(value.get("internal_force_assembly_elapsed_seconds", value.get("internal_force_elapsed_seconds"))),
        "effective_stiffness_assembly_elapsed_seconds": _optional_number(value.get("effective_stiffness_assembly_elapsed_seconds")),
        "coupled_assembly_elapsed_seconds": _optional_number(value.get("coupled_assembly_elapsed_seconds")),
        "monolithic_assembly_elapsed_seconds": _optional_number(value.get("monolithic_assembly_elapsed_seconds")),
        "reduced_matrix_elapsed_seconds": _optional_number(value.get("reduced_matrix_elapsed_seconds")),
        "augmented_bmat_elapsed_seconds": _optional_number(value.get("augmented_bmat_elapsed_seconds")),
        "lagrange_constraint_matrix_elapsed_seconds": _optional_number(value.get("lagrange_constraint_matrix_elapsed_seconds")),
        "lagrange_bmat_elapsed_seconds": _optional_number(value.get("lagrange_bmat_elapsed_seconds")),
        "lagrange_linear_solve_elapsed_seconds": _optional_number(value.get("lagrange_linear_solve_elapsed_seconds")),
        "linear_solve_elapsed_seconds": _optional_number(value.get("linear_solve_elapsed_seconds")),
        "line_search_elapsed_seconds": _optional_number(value.get("line_search_elapsed_seconds")),
        "postprocess_elapsed_seconds": _optional_number(value.get("postprocess_elapsed_seconds")),
        "residual_norm": "" if residual is None else residual,
        "converged": value.get("converged", ""),
        "linear_method": value.get("linear_method", ""),
        "reduced_matrix_cache_reused": value.get("reduced_matrix_cache_reused", ""),
        "reduced_matrix_cache_built": value.get("reduced_matrix_cache_built", ""),
        "symbolic_cache_state": value.get("symbolic_cache_state", ""),
        "lu_factor_cache_state": value.get("lu_factor_cache_state", ""),
    }


def _slowest_iteration(entries: list[dict[str, Any]]) -> dict[str, Any]:
    timed = [entry for entry in entries if _number_or_none(entry.get("elapsed_seconds")) is not None]
    if not timed:
        return {}
    return max(timed, key=lambda entry: _number_or_none(entry.get("elapsed_seconds")) or 0.0)


def _first_numeric(value: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _number_or_none(value.get(key))
        if number is not None:
            return number
    return None


def _optional_number(value: Any) -> float | str:
    number = _number_or_none(value)
    return "" if number is None else number


def _number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_array_bytes(result: SolveResult2D) -> int:
    total = 0
    for stage in result.stages:
        total += int(stage.displacements.nbytes + stage.reactions.nbytes)
        if isinstance(stage.pore_pressure, np.ndarray):
            total += int(stage.pore_pressure.nbytes)
    return total


def _estimated_sparse_bytes(size: int, nnz: int) -> int:
    return int(nnz * (8 + 4) + (size + 1) * 4)


def _performance_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for row in summary.get("stages", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{html.escape(str(row.get('method', '')))}</td>"
            f"<td>{html.escape(str(row.get('elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('matrix_size', '')))}</td>"
            f"<td>{html.escape(str(row.get('matrix_nnz', '')))}</td>"
            f"<td>{html.escape(str(row.get('solver_iterations', '')))}</td>"
            f"<td>{html.escape(str(row.get('dominant_category', '')))}</td>"
            f"<td>{html.escape(str(row.get('assembly_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('nonlinear_iteration_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('linear_solve_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('postprocess_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('coupled_assembly_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('srm_trial_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('cache_reuse_count', '')))}</td>"
            f"<td>{html.escape(str(row.get('cache_build_count', '')))}</td>"
            f"<td>{html.escape(str(row.get('cache_unused_reasons', '')))}</td>"
            f"<td>{html.escape(str(row.get('slowest_iteration_path', '')))}</td>"
            f"<td>{html.escape(str(row.get('slowest_iteration_elapsed_seconds', '')))}</td>"
            "</tr>"
        )
    profile = summary.get("profile", {}) if isinstance(summary.get("profile", {}), Mapping) else {}
    slowest_stage = profile.get("slowest_stage", {}) if isinstance(profile.get("slowest_stage", {}), Mapping) else {}
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM performance</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>Performance Summary</h1>
<p>elapsed={summary.get('elapsed_seconds', 0.0)}, stage_elapsed={summary.get('stage_elapsed_seconds', 0.0)}, stages={summary.get('stage_count', 0)}, iterations={summary.get('total_solver_iterations', 0)}</p>
<p>dominant category={html.escape(str(summary.get('dominant_category', '')))}, slowest stage={html.escape(str(slowest_stage.get('stage', '')))} ({html.escape(str(slowest_stage.get('elapsed_seconds', '')))}s), cache reuse={html.escape(str(summary.get('cache_reuse_count', 0)))}, cache build={html.escape(str(summary.get('cache_build_count', 0)))}</p>
<table><thead><tr><th>stage</th><th>method</th><th>elapsed</th><th>matrix size</th><th>nnz</th><th>iterations</th><th>dominant</th><th>assembly</th><th>nonlinear</th><th>linear solve</th><th>postprocess</th><th>coupled assembly</th><th>SRM trial</th><th>cache reuse</th><th>cache build</th><th>cache unused reasons</th><th>slowest iteration</th><th>slowest iteration elapsed</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def _benchmark_performance_html(report: Mapping[str, Any]) -> str:
    rows = []
    for row in report.get("cases", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('case', '')))}</td>"
            f"<td>{html.escape(str(row.get('category', '')))}</td>"
            f"<td>{html.escape(str(row.get('elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('stage_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('solve_elapsed_seconds_excluding_io', '')))}</td>"
            f"<td>{html.escape(str(row.get('cold_io_report_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('max_matrix_nnz', '')))}</td>"
            f"<td>{html.escape(str(row.get('total_solver_iterations', '')))}</td>"
            f"<td>{html.escape(str(row.get('dominant_category', '')))}</td>"
            f"<td>{html.escape(str(row.get('assembly_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('nonlinear_iteration_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('linear_solve_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('postprocess_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('coupled_assembly_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('srm_trial_elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(row.get('cache_reuse_count', '')))}</td>"
            f"<td>{html.escape(str(row.get('cache_build_count', '')))}</td>"
            f"<td>{html.escape(str(row.get('cache_unused_reasons', '')))}</td>"
            f"<td>{html.escape(str(row.get('slowest_iteration_path', '')))}</td>"
            f"<td>{html.escape(str(row.get('slowest_iteration_elapsed_seconds', '')))}</td>"
            "</tr>"
        )
    regressions = "".join(
        "<li>"
        + html.escape(f"{row.get('case')}: {row.get('metric')} current={row.get('current')} baseline={row.get('baseline')} slowdown={row.get('slowdown')}")
        + "</li>"
        for row in report.get("regressions", [])
        if isinstance(row, Mapping)
    )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM benchmark performance</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>Benchmark Performance</h1>
<p>cases={report.get('case_count', 0)}, regressions={report.get('regression_count', 0)}</p>
<table><thead><tr><th>case</th><th>category</th><th>cold elapsed</th><th>stage elapsed</th><th>solve excl. I/O</th><th>I/O/report</th><th>max nnz</th><th>iterations</th><th>dominant</th><th>assembly</th><th>nonlinear</th><th>linear solve</th><th>postprocess</th><th>coupled assembly</th><th>SRM trial</th><th>cache reuse</th><th>cache build</th><th>cache unused reasons</th><th>slowest iteration</th><th>slowest iteration elapsed</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Regressions</h2><ul>{regressions}</ul>
</body></html>
"""


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "PROFILE_CATEGORIES",
    "build_performance_summary",
    "write_performance_summary",
    "benchmark_case_performance",
    "compare_performance_cases",
    "write_benchmark_performance_reports",
]
