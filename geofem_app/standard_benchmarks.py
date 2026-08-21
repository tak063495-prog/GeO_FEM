"""GeoFEM-native benchmark suite independent of GeoFEAS compatibility."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from .fem2d_elements import (
    _quad4_axisymmetric_element_stiffness_fast,
    _quad4_axisymmetric_j2dp_tangent_force_fast,
    _quad4_element_stiffness_fast,
    _quad4_j2dp_tangent_force_fast,
    _quad8_axisymmetric_element_stiffness_fast,
    _quad8_axisymmetric_j2dp_tangent_force_fast,
    _quad8_element_stiffness_fast,
    _quad8_j2dp_tangent_force_fast,
)
from .fem2d_config import plane_strain_materials
from .fem2d_mesh import mesh_from_config
from .fem2d_materials import _yield_surface_parameters, principal_stresses
from .numba_warmup import benchmark_numba_warmup_enabled, skipped_numba_warmup_summary, warmup_numba_kernels
from .fem2d_performance_contract import large_deformation_fast_path_matrix
from .fem2d_solver import solve_plane_strain_config
from .fem2d_structural_assembly import assemble_global_stiffness_cached, build_global_stiffness_assembly_cache
from .fem2d_types import ElasticPlaneStrainMaterial, SolveResult2D, normalize_integration
from .performance_monitor import benchmark_case_performance, write_benchmark_performance_reports
from .samples import plane_strain_patch_sample, plane_strain_quad4_sample
from .sparse_assembly import (
    reset_sparse_assembly_diagnostics,
    set_sparse_assembly_diagnostics_enabled,
    sparse_assembly_diagnostics,
)


@dataclass(frozen=True)
class StandardBenchmarkCheck:
    name: str
    expected: float | str | bool | None
    actual: float | str | bool | None
    passed: bool
    tolerance: str = ""
    detail: str = ""


@dataclass(frozen=True)
class StandardBenchmarkCase:
    name: str
    category: str
    description: str
    config: dict[str, Any]
    validator: Callable[[SolveResult2D], list[StandardBenchmarkCheck]]
    expected: dict[str, Any] = field(default_factory=dict)


def build_standard_benchmark_cases() -> list[StandardBenchmarkCase]:
    return [
        _patch_case("QUAD4", "FULL"),
        _patch_case("TRI3", "FULL"),
        _cantilever_case(),
        _consolidation_case(),
        _axisymmetric_case(),
        _dynamic_case(),
        _nonlinear_static_case(),
        _riks_mpc_lagrange_case(),
        _axisymmetric_up_lagrange_case(),
        _nonlinear_srm_case(),
        _large_deformation_plastic_case("drucker_prager"),
        _large_deformation_plastic_case("mohr_coulomb"),
        _large_deformation_srm_case(),
        _large_deformation_quad8_case("FULL"),
        _large_deformation_quad8_case("SRI"),
        _large_deformation_quad8_case("B-BAR"),
        _large_deformation_hydro_case(),
    ]


def run_standard_benchmark_suite(
    output_dir: str | Path,
    *,
    baseline: Mapping[str, Any] | str | Path | None = None,
    max_slowdown: float = 1.25,
    max_structure_growth: float | None = None,
    max_memory_growth: float | None = None,
    max_fallback_increase: int = 0,
    update_baseline: bool = False,
    numba_warmup: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = build_standard_benchmark_cases()
    suite_started = time.perf_counter()
    if numba_warmup and benchmark_numba_warmup_enabled():
        warmup_summary = warmup_numba_kernels(profile="benchmark")
    else:
        warmup_summary = skipped_numba_warmup_summary(profile="benchmark", reason="disabled")
    (root / "standard_benchmark_numba_warmup.json").write_text(json.dumps(warmup_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    cases_started = time.perf_counter()
    case_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    baseline_data = _load_baseline(baseline)
    for case in cases:
        case_dir = root / case.name
        set_sparse_assembly_diagnostics_enabled(True)
        reset_sparse_assembly_diagnostics()
        t0 = time.perf_counter()
        try:
            result = solve_plane_strain_config(case.config, case_dir)
            elapsed = time.perf_counter() - t0
            sparse_diagnostics = sparse_assembly_diagnostics()
        finally:
            set_sparse_assembly_diagnostics_enabled(False)
        checks = case.validator(result)
        passed = all(check.passed for check in checks)
        performance_row = benchmark_case_performance(case.name, case.category, result, elapsed, passed=passed)
        performance_row.update(
            {
                "sparse_pattern_build_count": int(sparse_diagnostics.get("pattern_build_count", 0) or 0),
                "sparse_pattern_assemble_count": int(sparse_diagnostics.get("pattern_assemble_count", 0) or 0),
                "sparse_duplicate_scatter_count": int(sparse_diagnostics.get("pattern_duplicate_scatter_count", 0) or 0),
                "sparse_builder_to_csr_count": int(sparse_diagnostics.get("builder_to_csr_count", 0) or 0),
                "sparse_builder_block_count": int(sparse_diagnostics.get("builder_block_count", 0) or 0),
                "sparse_builder_value_count": int(sparse_diagnostics.get("builder_value_count", 0) or 0),
                "sparse_builder_elapsed_seconds": float(sparse_diagnostics.get("builder_elapsed_seconds", 0.0) or 0.0),
            }
        )
        summary = {
            "name": case.name,
            "category": case.category,
            "description": case.description,
            "expected": case.expected,
            "output_dir": str(result.output_dir),
            "elapsed_seconds": elapsed,
            "solve_elapsed_seconds_excluding_io": performance_row.get("solve_elapsed_seconds_excluding_io", 0.0),
            "io_report_elapsed_seconds": performance_row.get("io_report_elapsed_seconds", 0.0),
            "stage_io_report_elapsed_seconds": performance_row.get("stage_io_report_elapsed_seconds", 0.0),
            "run_io_report_elapsed_seconds": performance_row.get("run_io_report_elapsed_seconds", 0.0),
            "passed": passed,
            "sparse_assembly": sparse_diagnostics,
            "failed_count": sum(0 if check.passed else 1 for check in checks),
            "check_count": len(checks),
            "example_outputs": {
                "summary": str(result.output_dir / "summary.json"),
                "result_view_index": str(result.output_dir / "result_view_index.html"),
                "report": str(result.output_dir / "calculation_report.html"),
            },
        }
        (result.output_dir / "standard_benchmark_case.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        case_rows.append(summary)
        performance_rows.append(performance_row)
        for check in checks:
            check_rows.append({"case": case.name, "category": case.category, **check.__dict__})
    cases_elapsed = time.perf_counter() - cases_started
    performance_report = write_benchmark_performance_reports(
        performance_rows,
        root,
        baseline=baseline_data,
        max_slowdown=max_slowdown,
        max_structure_growth=max_structure_growth,
        max_memory_growth=max_memory_growth,
        max_fallback_increase=max_fallback_increase,
        update_baseline=update_baseline,
    )
    performance_report = _attach_numba_warmup_report(performance_report, root, warmup_summary)
    performance_report = _attach_quad8_scaling_benchmarks(performance_report, root)
    performance_report = _attach_real_mesh_scaling_benchmarks(performance_report, root)
    performance_report = _attach_large_deformation_fast_path_matrix(performance_report, root)
    suite = {
        "schema": "geofem.standard_benchmark_suite.v1",
        "case_count": len(case_rows),
        "passed": all(row["passed"] for row in case_rows) and bool(performance_report.get("passed", True)),
        "failed_count": sum(int(row["failed_count"]) for row in case_rows),
        "check_count": len(check_rows),
        "elapsed_seconds": time.perf_counter() - suite_started,
        "case_elapsed_seconds_excluding_warmup": cases_elapsed,
        "case_solve_elapsed_seconds_excluding_io": sum(float(row.get("solve_elapsed_seconds_excluding_io", 0.0) or 0.0) for row in performance_rows),
        "case_io_report_elapsed_seconds": sum(float(row.get("io_report_elapsed_seconds", 0.0) or 0.0) for row in performance_rows),
        "numba_warmup_elapsed_seconds": float(warmup_summary.get("elapsed_seconds", 0.0) or 0.0),
        "numba_warmup": warmup_summary,
        "performance_regression_count": int(performance_report.get("regression_count", 0) or 0),
        "performance": performance_report.get("paths", {}),
        "cases": case_rows,
    }
    (root / "standard_benchmark_summary.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(check_rows, root / "standard_benchmark_checks.csv")
    _write_html(suite, check_rows, root / "standard_benchmark_report.html")
    return suite


def _attach_numba_warmup_report(report: Mapping[str, Any], output_dir: Path, warmup_summary: Mapping[str, Any]) -> dict[str, Any]:
    paths = dict(report.get("paths", {})) if isinstance(report.get("paths", {}), Mapping) else {}
    paths["numba_warmup_json"] = str(output_dir / "standard_benchmark_numba_warmup.json")
    enriched = {
        **dict(report),
        "paths": paths,
        "numba_warmup_schema": warmup_summary.get("schema", "geofem.numba_warmup.v1"),
        "benchmark_numba_warmup_enabled": bool(warmup_summary.get("enabled", False)),
        "benchmark_numba_warmup_elapsed_seconds": float(warmup_summary.get("elapsed_seconds", 0.0) or 0.0),
        "benchmark_numba_warmup_kernel_count": int(warmup_summary.get("kernel_count", 0) or 0),
        "benchmark_numba_warmup_failed_count": int(warmup_summary.get("failed_count", 0) or 0),
        "benchmark_numba_warmup": dict(warmup_summary),
    }
    json_path = Path(paths.get("json", output_dir / "standard_benchmark_performance.json"))
    json_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return enriched


def _load_baseline(baseline: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any] | None:
    if baseline is None:
        return None
    if isinstance(baseline, Mapping):
        return baseline
    path = Path(baseline)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else None


def _attach_quad8_scaling_benchmarks(report: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    rows = _quad8_scaling_benchmark_rows()
    csv_path = output_dir / "standard_benchmark_quad8_scaling.csv"
    _write_quad8_scaling_csv(rows, csv_path)
    paths = dict(report.get("paths", {})) if isinstance(report.get("paths", {}), Mapping) else {}
    paths["quad8_scaling_csv"] = str(csv_path)
    warmup_elapsed = sum(float(row.get("numba_warmup_elapsed_seconds", 0.0) or 0.0) for row in rows)
    warm_elapsed = sum(float(row.get("warm_elapsed_seconds", 0.0) or 0.0) for row in rows)
    enriched = {
        **dict(report),
        "paths": paths,
        "quad8_scaling_schema": "geofem.quad8_scaling_benchmark.v1",
        "quad8_scaling_count": len(rows),
        "numba_kernel_warmup_elapsed_seconds": warmup_elapsed,
        "numba_kernel_warm_elapsed_seconds": warm_elapsed,
        "numba_kernel_cold_warm_ratio": warmup_elapsed / warm_elapsed if warm_elapsed > 0.0 else "",
        "quad8_scaling": rows,
    }
    json_path = Path(paths.get("json", output_dir / "standard_benchmark_performance.json"))
    json_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return enriched


def _attach_large_deformation_fast_path_matrix(report: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    matrix = large_deformation_fast_path_matrix()
    json_path = output_dir / "large_deformation_fast_path_matrix.json"
    csv_path = output_dir / "large_deformation_fast_path_matrix.csv"
    json_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_matrix_csv(matrix["rows"], csv_path)
    paths = dict(report.get("paths", {})) if isinstance(report.get("paths", {}), Mapping) else {}
    paths["large_deformation_fast_path_matrix_json"] = str(json_path)
    paths["large_deformation_fast_path_matrix_csv"] = str(csv_path)
    enriched = {**dict(report), "paths": paths, "large_deformation_fast_path_matrix": str(json_path)}
    perf_json = Path(paths.get("json", output_dir / "standard_benchmark_performance.json"))
    perf_json.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return enriched


def _attach_real_mesh_scaling_benchmarks(report: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    rows = _real_mesh_scaling_benchmark_rows()
    gates: list[dict[str, Any]] = []
    for element_type in ("QUAD4", "QUAD8"):
        group = [row for row in rows if row["element_type"] == element_type]
        group.sort(key=lambda row: int(row["element_count"]))
        first = group[0]
        last = group[-1]
        normalized_growth = float(last["warm_seconds_per_element"]) / max(
            float(first["warm_seconds_per_element"]),
            np.finfo(float).eps,
        )
        fully_batched = all(int(row["batched_element_count"]) == int(row["element_count"]) for row in group)
        finite = all(math.isfinite(float(row["warm_elapsed_seconds"])) and float(row["warm_elapsed_seconds"]) >= 0.0 for row in group)
        passed = bool(fully_batched and finite and normalized_growth <= 8.0)
        gates.append(
            {
                "element_type": element_type,
                "passed": passed,
                "fully_batched": fully_batched,
                "finite_timings": finite,
                "small_element_count": int(first["element_count"]),
                "large_element_count": int(last["element_count"]),
                "normalized_per_element_growth": normalized_growth,
                "max_normalized_per_element_growth": 8.0,
            }
        )
    payload = {
        "schema": "geofem.real_mesh_scaling_benchmark.v1",
        "passed": all(bool(gate["passed"]) for gate in gates),
        "row_count": len(rows),
        "max_element_count": max((int(row["element_count"]) for row in rows), default=0),
        "gates": gates,
        "rows": rows,
    }
    json_path = output_dir / "standard_benchmark_real_mesh_scaling.json"
    csv_path = output_dir / "standard_benchmark_real_mesh_scaling.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(rows, csv_path)
    paths = dict(report.get("paths", {})) if isinstance(report.get("paths", {}), Mapping) else {}
    paths["real_mesh_scaling_json"] = str(json_path)
    paths["real_mesh_scaling_csv"] = str(csv_path)
    failed_gate_count = sum(1 for gate in gates if not bool(gate["passed"]))
    enriched = {
        **dict(report),
        "passed": bool(report.get("passed", True)) and bool(payload["passed"]),
        "regression_count": int(report.get("regression_count", 0) or 0) + failed_gate_count,
        "paths": paths,
        "real_mesh_scaling_schema": payload["schema"],
        "real_mesh_scaling_passed": payload["passed"],
        "real_mesh_scaling_gate_count": len(gates),
        "real_mesh_scaling_failed_gate_count": failed_gate_count,
        "real_mesh_scaling_max_element_count": payload["max_element_count"],
        "real_mesh_scaling": payload,
    }
    perf_json = Path(paths.get("json", output_dir / "standard_benchmark_performance.json"))
    perf_json.write_text(json.dumps(enriched, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return enriched


def _real_mesh_scaling_benchmark_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workloads = {
        "QUAD4": ((16, 8), (32, 16), (64, 32)),
        "QUAD8": ((8, 4), (16, 8), (32, 16)),
    }
    for element_type, sizes in workloads.items():
        for nx, ny in sizes:
            cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain"},
                "mesh": {
                    "generator": "rectangle",
                    "x_range": [0.0, float(nx)],
                    "y_range": [0.0, float(ny)],
                    "nx": nx,
                    "ny": ny,
                    "element_type": element_type,
                    "integration": "FULL",
                    "material": "soil",
                },
                "materials": {"soil": {"model": "elastic", "E": 50000.0, "nu": 0.3}},
            }
            mesh = mesh_from_config(cfg)
            materials = plane_strain_materials(cfg)
            cache_started = time.perf_counter()
            cache = build_global_stiffness_assembly_cache(mesh, materials)
            cache_elapsed = max(time.perf_counter() - cache_started, 0.0)
            assemble_global_stiffness_cached(cache, mesh, materials)
            timings: list[float] = []
            matrix_nnz = 0
            for _repeat in range(3):
                started = time.perf_counter()
                matrix = assemble_global_stiffness_cached(cache, mesh, materials)
                timings.append(max(time.perf_counter() - started, 0.0))
                matrix_nnz = int(matrix.nnz)
            warm_elapsed = float(np.median(np.asarray(timings, dtype=float)))
            element_count = len(mesh.elements)
            info = cache.info()
            rows.append(
                {
                    "benchmark": "real_mesh_global_stiffness_scaling",
                    "element_type": element_type,
                    "integration": "FULL",
                    "nx": nx,
                    "ny": ny,
                    "node_count": len(mesh.node_ids),
                    "element_count": element_count,
                    "dof_count": int(matrix.shape[0]),
                    "matrix_nnz": matrix_nnz,
                    "cache_build_elapsed_seconds": cache_elapsed,
                    "warm_elapsed_seconds": warm_elapsed,
                    "warm_seconds_per_element": warm_elapsed / max(element_count, 1),
                    "batched_element_count": int(info.get("batched_elastic_elements", 0) or 0),
                    "batch_coverage_ratio": int(info.get("batched_elastic_elements", 0) or 0) / max(element_count, 1),
                    "direct_fill_enabled": bool(info.get("direct_fill", {}).get("enabled", False)) if isinstance(info.get("direct_fill", {}), Mapping) else False,
                }
            )
    return rows


def _quad8_scaling_benchmark_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = ("FULL", "SRI", "B-BAR")
    densities = (1, 2, 4)
    for density in densities:
        element_count = density * density
        for analysis_type in ("plane_strain", "axisymmetric"):
            for element_type in ("QUAD4", "QUAD8"):
                available_modes = modes if element_type == "QUAD8" or analysis_type == "plane_strain" else ("FULL",)
                for mode in available_modes:
                    _append_quad_scaling_row(rows, analysis_type, element_type, mode, "elastic", density, element_count)
            for element_type in ("QUAD4", "QUAD8"):
                available_modes = modes if element_type == "QUAD8" else ("FULL",)
                for mode in available_modes:
                    _append_quad_scaling_row(rows, analysis_type, element_type, mode, "j2", density, element_count)
    return rows


def _append_quad_scaling_row(
    rows: list[dict[str, Any]],
    analysis_type: str,
    element_type: str,
    mode: str,
    material_model: str,
    density: int,
    element_count: int,
) -> None:
    kernel, state_point_count = _quad_scaling_kernel(analysis_type, element_type, mode, material_model)
    kernel_name = f"{analysis_type}_{element_type.lower()}_{mode.lower().replace('-', '_')}_{material_model}"
    warmup_elapsed, warm_elapsed = _time_warm_kernel(kernel, element_count)
    rows.append(
        {
            "benchmark": "quad8_kernel_scaling",
            "scope": "single_element_kernel_repeated_as_same_density_mesh",
            "comparison_group": f"{analysis_type}:{mode}:{material_model}:{density}x{density}",
            "analysis_type": analysis_type,
            "element_type": element_type,
            "integration": mode,
            "material_model": material_model,
            "mesh_density": f"{density}x{density}",
            "element_count": element_count,
            "state_point_count": state_point_count,
            "numba_warmup_elapsed_seconds": warmup_elapsed,
            "warm_elapsed_seconds": warm_elapsed,
            "per_element_warm_seconds": warm_elapsed / max(element_count, 1),
            "kernel": kernel_name,
        }
    )


def _quad_scaling_kernel(
    analysis_type: str,
    element_type: str,
    mode: str,
    material_model: str,
) -> tuple[Callable[[], None], int]:
    coords4, coords8 = _benchmark_coords(analysis_type)
    if material_model == "elastic":
        material = ElasticPlaneStrainMaterial("bench", E=50000.0, nu=0.30)
        if analysis_type == "axisymmetric" and element_type == "QUAD4":
            return lambda: _discard(_quad4_axisymmetric_element_stiffness_fast(coords4, material)), 4
        if analysis_type == "axisymmetric":
            return lambda: _discard(_quad8_axisymmetric_element_stiffness_fast(coords8, material, mode)), 9 if mode != "SRI" else 13
        if element_type == "QUAD4":
            return lambda: _discard(_quad4_element_stiffness_fast(coords4, material, mode)), 4
        return lambda: _discard(_quad8_element_stiffness_fast(coords8, material, mode)), 9 if mode != "SRI" else 13

    material = ElasticPlaneStrainMaterial("bench", E=50000.0, nu=0.30, model="von_mises", yield_stress=80.0)
    alpha, cohesion_term = _yield_surface_parameters(material, 1.0)
    initial = np.zeros(4, dtype=float)
    if element_type == "QUAD4":
        ue = _benchmark_displacements(4)
        plastic = np.zeros((4, 4), dtype=float)
        kappas = np.zeros(4, dtype=float)
        if analysis_type == "axisymmetric":
            return (
                lambda: _discard(
                    _quad4_axisymmetric_j2dp_tangent_force_fast(
                        coords4,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    )
                ),
                4,
            )
        return (
            lambda: _discard(
                _quad4_j2dp_tangent_force_fast(
                    coords4,
                    ue,
                    material,
                    initial_stress=initial,
                    plastic_strains=plastic,
                    kappas=kappas,
                    alpha=alpha,
                    cohesion_term=cohesion_term,
                )
            ),
            4,
        )

    ue = _benchmark_displacements(8)
    point_count = 13 if mode == "SRI" else 9
    plastic = np.zeros((point_count, 4), dtype=float)
    kappas = np.zeros(point_count, dtype=float)
    if analysis_type == "axisymmetric":
        return (
            lambda: _discard(
                _quad8_axisymmetric_j2dp_tangent_force_fast(
                    coords8,
                    ue,
                    material,
                    mode,
                    initial_stress=initial,
                    plastic_strains=plastic,
                    kappas=kappas,
                    alpha=alpha,
                    cohesion_term=cohesion_term,
                )
            ),
            point_count,
        )
    return (
        lambda: _discard(
            _quad8_j2dp_tangent_force_fast(
                coords8,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
        ),
        point_count,
    )


def _benchmark_coords(analysis_type: str) -> tuple[np.ndarray, np.ndarray]:
    x0 = 1.0 if analysis_type == "axisymmetric" else 0.0
    x1 = x0 + 1.0
    coords4 = np.array([[x0, 0.0], [x1, 0.0], [x1, 1.0], [x0, 1.0]], dtype=float)
    coords8 = np.array(
        [
            [x0, 0.0],
            [x1, 0.0],
            [x1, 1.0],
            [x0, 1.0],
            [(x0 + x1) * 0.5, 0.0],
            [x1, 0.5],
            [(x0 + x1) * 0.5, 1.0],
            [x0, 0.5],
        ],
        dtype=float,
    )
    return coords4, coords8


def _benchmark_displacements(node_count: int) -> np.ndarray:
    values = np.zeros(node_count * 2, dtype=float)
    for node in range(node_count):
        x_scale = 1.0 + 0.1 * node
        y_scale = 1.0 - 0.05 * node
        values[2 * node] = 1.0e-4 * x_scale
        values[2 * node + 1] = -7.0e-5 * y_scale
    return values


def _time_warm_kernel(kernel: Callable[[], None], repetitions: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    for _ in range(repetitions):
        kernel()
    warmup_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(repetitions):
        kernel()
    warm_elapsed = time.perf_counter() - t1
    return warmup_elapsed, warm_elapsed


def _discard(value: Any) -> None:
    _ = value


def _write_quad8_scaling_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "benchmark",
        "scope",
        "comparison_group",
        "analysis_type",
        "element_type",
        "integration",
        "material_model",
        "mesh_density",
        "element_count",
        "state_point_count",
        "numba_warmup_elapsed_seconds",
        "warm_elapsed_seconds",
        "per_element_warm_seconds",
        "kernel",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_matrix_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fields = ["deformation_mode", "element_type", "integration", "material_model", "tension_cutoff", "hydro_coupled", "status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _patch_case(element: str, integration: str) -> StandardBenchmarkCase:
    cfg = plane_strain_patch_sample(element, integration)
    mat = ElasticPlaneStrainMaterial("soil", E=50000.0, nu=0.30)
    expected_strain = np.array([0.001, -0.0003, 0.0, 0.0006])
    expected_stress = mat.D4 @ expected_strain

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        row = result.stages[0].element_results[0]
        actual_strain = np.array([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], dtype=float)
        actual_stress = np.array([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], dtype=float)
        return [
            _check_close("strain_patch", 0.0, float(np.linalg.norm(actual_strain - expected_strain)), atol=1.0e-10),
            _check_close("stress_patch", 0.0, float(np.linalg.norm(actual_stress - expected_stress)), atol=1.0e-6),
            _check_bool("principal_order", True, principal_stresses(actual_stress)[0] >= principal_stresses(actual_stress)[1]),
        ]

    return StandardBenchmarkCase(
        name=f"patch_{element.lower()}_{integration.lower().replace('-', '_')}",
        category="patch_test",
        description=f"{element} {integration} の線形変位パッチテスト",
        config=cfg,
        validator=validate,
        expected={"strain": expected_strain.tolist(), "stress": expected_stress.tolist()},
    )


def _cantilever_case() -> StandardBenchmarkCase:
    cfg = plane_strain_quad4_sample(integration="B-bar")

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        max_u = max(float(np.hypot(stage.displacements[i], stage.displacements[i + 1])) for i in range(0, len(stage.displacements), 2))
        return [
            _check_bool("finite_displacements", True, bool(np.all(np.isfinite(stage.displacements)))),
            _check_bool("nonzero_response", True, max_u > 0.0),
            _check_bool("result_view_index", True, (result.output_dir / "result_view_index.json").exists()),
        ]

    return StandardBenchmarkCase("cantilever_quad4_bbar", "static", "片持ち梁風の静的弾性応答", cfg, validate, expected={"response": "finite nonzero displacement"})


def _consolidation_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
        "mesh": {"generator": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "boundary_conditions": [{"set": "all", "fixed": True}],
        "steps": [{"name": "consolidation", "type": "consolidation", "hydro": {"initial_pressure": 100.0, "dt": 0.1, "steps": 2, "storage": 1.0, "permeability": 1.0, "pressure_bcs": [{"set": "top", "pressure": 0.0}]}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        pressure = result.stages[0].pore_pressure
        assert pressure is not None
        return [
            _check_bool("top_drained", True, bool(np.allclose(pressure[[2, 3]], 0.0))),
            _check_bool("bottom_dissipates", True, bool(np.all((pressure[[0, 1]] > 0.0) & (pressure[[0, 1]] < 100.0)))),
        ]

    return StandardBenchmarkCase("consolidation_diffusion", "consolidation", "一次元排水境界を持つ圧密拡散", cfg, validate, expected={"top_pressure": 0.0, "bottom_pressure": "0 < p < 100"})


def _axisymmetric_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
        "mesh": {"generator": "rectangle", "x_range": [1.0, 2.0], "y_range": [0.0, 2.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 10.0}},
        "boundary_conditions": [{"set": "all", "fixed": True}],
        "steps": [{"name": "axisym-k0", "type": "geostatic", "apply_gravity": False, "surface_y": 2.0, "k0": 0.4}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        row = result.stages[0].element_results[0]
        return [
            _check_close("sigma_y", -10.0, float(row["sigma_y"]), atol=1.0e-10),
            _check_close("sigma_z", -4.0, float(row["sigma_z"]), atol=1.0e-10),
        ]

    return StandardBenchmarkCase("axisymmetric_k0", "axisymmetric", "軸対称K0初期応力", cfg, validate, expected={"sigma_y": -10.0, "sigma_z": -4.0})


def _dynamic_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]}, "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}], "node_sets": {"base": ["1", "2"]}},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
        "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
        "steps": [{"name": "dynamic", "type": "dynamic_time_history", "dynamic": {"rayleigh_alpha": 0.02}, "seismic": {"time_history": [{"time": 0.0, "kh": 0.0}, {"time": 0.1, "kh": 0.10}, {"time": 0.2, "kh": 0.0}]}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        return [
            _check_bool("newmark_method", True, stage.solver_info.get("method") == "newmark"),
            _check_bool("dynamic_history", True, (stage.output_dir / "dynamic_history.csv").exists()),
        ]

    return StandardBenchmarkCase("dynamic_newmark", "dynamic", "Newmark時刻歴応答", cfg, validate, expected={"method": "newmark", "history_file": "dynamic_history.csv"})


def _nonlinear_static_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {
            "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
            "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
        },
        "materials": {"soil": {"model": "von_mises", "E": 10000.0, "nu": 0.30, "yield_stress": 1.0, "hardening": 100.0}},
        "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
        "loads": [{"node": "3", "fy": -10.0}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        return [
            _check_bool("newton_method", True, stage.solver_info.get("method") == "newton"),
            _check_bool("converged", True, bool(stage.solver_info.get("converged", False))),
            _check_bool("convergence_history_profiled", True, isinstance(stage.solver_info.get("convergence_history"), list) and len(stage.solver_info["convergence_history"]) >= 2),
            _check_bool("finite_response", True, bool(np.all(np.isfinite(stage.displacements)))),
        ]

    return StandardBenchmarkCase(
        "nonlinear_static_von_mises",
        "nonlinear_profile",
        "通常非線形静解析のNewton反復プロファイル代表ケース",
        cfg,
        validate,
        expected={"method": "newton", "profile": "convergence_history"},
    )


def _riks_mpc_lagrange_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {
            "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
            "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
        },
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
        "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
        "steps": [{"name": "riks-lm", "type": "riks", "riks": {"lambda_max": 2.0, "steps": 2}, "loads": [{"node": "3", "fy": -10.0}]}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        riks = stage.solver_info.get("riks", {})
        mpc = stage.solver_info.get("mpc", {})
        performance = stage.solver_info.get("performance", {}) if isinstance(stage.solver_info.get("performance", {}), Mapping) else {}
        return [
            _check_bool("arc_length_method", True, stage.solver_info.get("method") == "arc_length"),
            _check_close("lambda", 2.0, float(riks.get("lambda", math.nan)), atol=1.0e-12),
            _check_bool("lagrange_mpc", True, mpc.get("applied_method") == "lagrange"),
            _check_bool("riks_path_profiled", True, isinstance(riks.get("path"), list) and len(riks["path"]) == 2),
            _check_bool("riks_lagrange_profiled", True, float(performance.get("lagrange_bmat_elapsed_seconds", 0.0) or 0.0) >= 0.0 and "lagrange_constraint_matrix_elapsed_seconds" in performance),
            _check_bool("riks_linear_profiled", True, float(performance.get("linear_solve_elapsed_seconds", 0.0) or 0.0) > 0.0),
        ]

    return StandardBenchmarkCase(
        "riks_mpc_lagrange",
        "arc_length_profile",
        "Riks/arc-lengthとLagrange MPCの性能プロファイル代表ケース",
        cfg,
        validate,
        expected={"method": "arc_length", "mpc": "lagrange"},
    )


def _axisymmetric_up_lagrange_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
        "mesh": {"generator": "rectangle", "x_range": [1.0, 2.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
        "mpc_constraints": [{"master": "2", "slave": "4", "dof": "ux", "coefficient": 1.0, "method": "lagrange"}],
        "steps": [
            {
                "name": "axisym-up-lm",
                "type": "consolidation",
                "hydro": {"dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0, "pressure_bcs": [{"set": "top", "pressure": 0.0}]},
                "loads": [{"edge": ["3", "4"], "ty": -1.0}],
            }
        ],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        pressure = stage.pore_pressure
        return [
            _check_bool("axisymmetric_up_method", True, stage.solver_info.get("method") == "axisymmetric_monolithic_up"),
            _check_bool("pressure_available", True, pressure is not None and bool(np.all(np.isfinite(pressure)))),
            _check_bool("lagrange_mpc", True, stage.solver_info.get("mpc", {}).get("applied_method") == "lagrange"),
            _check_bool("consolidation_step_profiled", True, isinstance(stage.solver_info.get("consolidation", {}).get("step_history"), list)),
        ]

    return StandardBenchmarkCase(
        "axisymmetric_up_lagrange",
        "axisymmetric_up_profile",
        "軸対称u-pとLagrange MPCの性能プロファイル代表ケース",
        cfg,
        validate,
        expected={"method": "axisymmetric_monolithic_up", "mpc": "lagrange"},
    )


def _nonlinear_srm_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {"generator": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
        "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 80.0}},
        "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
        "loads": [{"edge": ["3", "4"], "ty": -10.0}],
        "steps": [{"name": "srm", "type": "srm", "srm": {"factors": [1.0, 2.0], "failure_plastic_ratio": 0.0}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        srm = result.stages[0].solver_info.get("srm", {})
        return [_check_close("factor_of_safety", 2.0, float(srm.get("factor_of_safety", math.nan)), atol=1.0e-12)]

    return StandardBenchmarkCase("nonlinear_srm_von_mises", "nonlinear", "von Mises材料の強度低減代表ケース", cfg, validate, expected={"factor_of_safety": 2.0})


def _large_deformation_plastic_case(model: str) -> StandardBenchmarkCase:
    mat: dict[str, Any]
    if model == "mohr_coulomb":
        mat = {"model": "mohr_coulomb", "E": 50000.0, "nu": 0.30, "cohesion": 500.0, "friction_angle": 30.0, "dilation_angle": 5.0}
    else:
        mat = {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 500.0, "friction_angle": 30.0, "dilation_angle": 5.0}
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {"generator": "rectangle", "x_range": [0.0, 3.0], "y_range": [0.0, 2.0], "nx": 3, "ny": 2, "element_type": "QUAD4", "integration": "FULL", "material": "soil"},
        "materials": {"soil": mat},
        "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}, {"set": "top", "uy": -0.001}],
        "steps": [{"name": f"large-{model}", "type": "large_deformation", "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        info = stage.solver_info
        large = info.get("large_deformation", {})
        return [
            _check_bool("updated_lagrangian", True, info.get("method") == "updated_lagrangian"),
            _check_bool("plastic_internal_loop", True, all(row.get("increment_solver") == "large_deformation_plastic_internal_loop" for row in large.get("history", []))),
            _check_bool("common_solver_info", True, all(key in info for key in ("geometry_mode", "element_type", "integration", "material_model", "batched_elements", "fallback_count", "fallback_reasons"))),
            _check_bool("final_only_postprocess", True, bool(large.get("history", [{}])[-1].get("postprocessed", False))),
        ]

    return StandardBenchmarkCase(
        f"large_deformation_quad4_{model}",
        "large_deformation",
        f"QUAD4 {model} 大変形の内部ループ・solver_info代表ケース",
        cfg,
        validate,
        expected={"increment_solver": "large_deformation_plastic_internal_loop"},
    )


def _large_deformation_srm_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {"generator": "rectangle", "x_range": [0.0, 3.0], "y_range": [0.0, 2.0], "nx": 3, "ny": 2, "element_type": "QUAD4", "integration": "FULL", "material": "soil"},
        "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 80.0}},
        "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}],
        "loads": [{"set": "top", "fy": -1.0}],
        "steps": [{"name": "large-srm", "type": "srm", "solver": {"large_deformation": {"enabled": True, "steps": 2, "adaptive_steps": False, "backend": "vectorized"}}, "srm": {"factors": [0.8, 1.0], "failure_plastic_ratio": 1.0}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        srm = stage.solver_info.get("srm", {})
        return [
            _check_bool("srm_result", True, "factor_of_safety" in srm),
            _check_bool("srm_strength_factor_reported", True, "strength_factor" in stage.solver_info),
            _check_bool("large_solver_info", True, stage.solver_info.get("method") == "updated_lagrangian"),
        ]

    return StandardBenchmarkCase("large_deformation_srm", "large_deformation", "SRM強度低減と大変形内部ループの結合代表ケース", cfg, validate, expected={"srm": "large_deformation"})


def _large_deformation_quad8_case(integration: str) -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
        "mesh": {
            "nodes": {
                "1": [0.0, 0.0],
                "2": [1.0, 0.0],
                "3": [1.0, 1.0],
                "4": [0.0, 1.0],
                "5": [0.5, 0.0],
                "6": [1.0, 0.5],
                "7": [0.5, 1.0],
                "8": [0.0, 0.5],
            },
            "elements": [{"id": "e1", "type": "QUAD8", "nodes": ["1", "2", "3", "4", "5", "6", "7", "8"], "material": "soil", "integration": integration}],
            "node_sets": {"base": ["1", "2", "5"], "top": ["3", "4", "7"]},
        },
        "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 500.0, "friction_angle": 30.0}},
        "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}, {"set": "top", "uy": -0.001}],
        "steps": [{"name": f"large-quad8-{integration}", "type": "large_deformation", "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        info = stage.solver_info
        return [
            _check_bool("quad8_element_type", True, info.get("element_type") == "QUAD8"),
            _check_bool("integration_reported", True, info.get("integration") == normalize_integration(integration)),
            _check_bool("finite_response", True, bool(np.all(np.isfinite(stage.displacements)))),
        ]

    return StandardBenchmarkCase(
        f"large_deformation_quad8_{integration.lower().replace('-', '_')}",
        "large_deformation",
        f"QUAD8 {integration} 大変形DP代表ケース",
        cfg,
        validate,
        expected={"element_type": "QUAD8", "integration": integration},
    )


def _large_deformation_hydro_case() -> StandardBenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
        "mesh": {"generator": "rectangle", "x_range": [0.0, 2.0], "y_range": [0.0, 2.0], "nx": 2, "ny": 2, "element_type": "QUAD4", "integration": "FULL", "material": "soil"},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "permeability": 1.0, "storage": 1.0, "biot": 1.0}},
        "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}, {"set": "top", "uy": -0.0005}],
        "steps": [{"name": "large-hydro", "type": "large_deformation", "hydro": {"initial_pressure": 10.0}, "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}],
    }

    def validate(result: SolveResult2D) -> list[StandardBenchmarkCheck]:
        stage = result.stages[0]
        return [
            _check_bool("pore_pressure_carried", True, stage.pore_pressure is not None),
            _check_bool("hydro_solver_info", True, bool(stage.solver_info.get("hydro_coupled", False))),
            _check_bool("up_cache", True, bool(stage.solver_info.get("large_deformation", {}).get("topology_cache", {}).get("hydro_cache", {}).get("enabled", False))),
        ]

    return StandardBenchmarkCase("large_deformation_hydro", "large_deformation", "間隙水圧入力付き大変形代表ケース", cfg, validate, expected={"hydro_coupled": True})


def _check_close(name: str, expected: float, actual: float, *, atol: float = 1.0e-8, rtol: float = 1.0e-6) -> StandardBenchmarkCheck:
    passed = math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected)
    return StandardBenchmarkCheck(name, expected, actual, passed, f"atol={atol}, rtol={rtol}")


def _check_bool(name: str, expected: bool, actual: bool) -> StandardBenchmarkCheck:
    return StandardBenchmarkCheck(name, expected, actual, bool(actual) is bool(expected))


def _write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    fields = ["case", "category", "name", "expected", "actual", "passed", "tolerance", "detail"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_html(summary: Mapping[str, Any], checks: list[Mapping[str, Any]], path: Path) -> None:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('case', '')))}</td>"
        f"<td>{html.escape(str(row.get('name', '')))}</td>"
        f"<td>{html.escape(str(row.get('expected', '')))}</td>"
        f"<td>{html.escape(str(row.get('actual', '')))}</td>"
        f"<td>{html.escape(str(row.get('passed', '')))}</td>"
        "</tr>"
        for row in checks
    )
    warmup_elapsed = float(summary.get("numba_warmup_elapsed_seconds", 0.0) or 0.0)
    case_elapsed = float(summary.get("case_elapsed_seconds_excluding_warmup", 0.0) or 0.0)
    path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM standard benchmark report</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>GeoFEM 標準ベンチマーク</h1>
<p>passed={bool(summary.get('passed', False))}, cases={int(summary.get('case_count', 0) or 0)}, failed={int(summary.get('failed_count', 0) or 0)}</p>
<p>numba_warmup={warmup_elapsed:.4f}s, cases_excluding_warmup={case_elapsed:.4f}s</p>
<table><thead><tr><th>case</th><th>check</th><th>expected</th><th>actual</th><th>passed</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>
""",
        encoding="utf-8",
    )


__all__ = ["build_standard_benchmark_cases", "run_standard_benchmark_suite"]
