"""GeoFEAS-like verification benchmark suite for 2D regression coverage."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

import numpy as np

from .fem2d_solver import solve_plane_strain_config
from .fem2d_types import SolveResult2D, StageResult2D
from .geofeas_public import PUBLIC_PROFILE, write_public_compatibility_matrix


@dataclass(frozen=True)
class BenchmarkCheck:
    name: str
    metric: str
    expected: float | str | bool | None = None
    actual: float | str | bool | None = None
    rtol: float = 1.0e-5
    atol: float = 1.0e-8
    passed: bool = True
    detail: str = ""


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    description: str
    config: dict[str, Any]
    validator: Callable[[SolveResult2D], list[BenchmarkCheck]]
    required_outputs: tuple[str, ...] = ("calculation_report.html", "calculation_report.pdf", "calculation_report_manifest.json", "calculation_report_input_snapshot.json")
    tags: tuple[str, ...] = field(default_factory=tuple)


def build_geofeas_benchmark_cases(work_dir: str | Path) -> list[BenchmarkCase]:
    """Return small GeoFEAS-like benchmark cases covering core workflows."""

    root = Path(work_dir)
    seepage_csv = root / "seepage_pressure_history.csv"
    seepage_csv.parent.mkdir(parents=True, exist_ok=True)
    seepage_csv.write_text("time,node_id,head\n0.0,1,1.0\n1.0,1,2.0\n1.0,2,2.0\n", encoding="utf-8")
    return [
        _thin_axisymmetric_case(),
        _srm_case(),
        _excavation_case(),
        _liquefaction_case(),
        _seepage_pressure_case(seepage_csv),
        _mpc_case(),
        _joint_case(),
    ]


def run_geofeas_benchmark_suite(output_dir: str | Path) -> dict[str, Any]:
    """Run the benchmark suite and write JSON/CSV/HTML verification artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = build_geofeas_benchmark_cases(root / "_inputs")
    case_summaries: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        case_root = root / case.name
        t0 = time.perf_counter()
        result = solve_plane_strain_config(case.config, case_root)
        elapsed = time.perf_counter() - t0
        checks = case.validator(result)
        checks.extend(_report_output_checks(result, case.required_outputs))
        appearance = _post_appearance_summary(result)
        checks.extend(_post_appearance_checks(result, appearance))
        passed = all(check.passed for check in checks)
        metrics = _benchmark_metrics(result)
        check_rows = [_check_to_row(case, check) for check in checks]
        all_checks.extend(check_rows)
        case_summary = {
            "name": case.name,
            "category": case.category,
            "description": case.description,
            "tags": list(case.tags),
            "output_dir": str(result.output_dir),
            "elapsed_seconds": elapsed,
            "passed": passed,
            "failed_count": sum(0 if check.passed else 1 for check in checks),
            "check_count": len(checks),
            "metrics": metrics,
            "post_appearance": appearance,
            "reports": {
                "html": str(result.output_dir / "calculation_report.html"),
                "pdf": str(result.output_dir / "calculation_report.pdf"),
                "manifest": str(result.output_dir / "calculation_report_manifest.json"),
            },
        }
        (result.output_dir / "benchmark_case_summary.json").write_text(json.dumps(case_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _write_case_tolerance_csv(check_rows, result.output_dir / "benchmark_tolerance.csv")
        case_summaries.append(case_summary)
    public_matrix = write_public_compatibility_matrix(
        root,
        cases=[{"name": case.name, "tags": list(case.tags), "config": case.config} for case in cases],
    )
    summary = {
        "schema": "geofem.geofeas_benchmark_suite.v1",
        "case_count": len(case_summaries),
        "passed": all(case["passed"] for case in case_summaries),
        "failed_count": sum(int(case["failed_count"]) for case in case_summaries),
        "check_count": len(all_checks),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": case_summaries,
        "public_compatibility": {
            "profile": public_matrix["profile"],
            "passed": public_matrix["passed"],
            "json": public_matrix["json"],
            "csv": public_matrix["csv"],
            "html": public_matrix["html"],
            "public_implemented_count": public_matrix["public_implemented_count"],
            "blocked_proprietary_count": public_matrix["blocked_proprietary_count"],
        },
    }
    (root / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_case_tolerance_csv(all_checks, root / "benchmark_tolerance.csv")
    _write_metrics_csv(case_summaries, root / "benchmark_metrics.csv")
    _write_benchmark_html(summary, all_checks, root / "benchmark_report.html")
    return summary


def _thin_axisymmetric_case() -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "axisymmetric_static", "profile": PUBLIC_PROFILE},
        "mesh": {
            "generator": "rectangle",
            "x_range": [1.0, 2.0],
            "y_range": [0.0, 2.0],
            "nx": 1,
            "ny": 1,
            "element_type": "QUAD4",
            "material": "soil",
        },
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 10.0}},
        "boundary_conditions": [{"set": "all", "fixed": True}],
        "steps": [{"name": "thin-axisym-k0", "type": "geostatic", "geofeas_workflow": "axisymmetric", "apply_gravity": False, "surface_y": 2.0, "k0": 0.4}],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        stage = result.stages[0]
        row = stage.element_results[0]
        return [
            _eq("axisymmetric.geometry", "geometry", stage.solver_info.get("geometry"), "axisymmetric"),
            _near("axisymmetric.sigma_y", "sigma_y", float(row["sigma_y"]), -10.0, atol=1.0e-10),
            _near("axisymmetric.sigma_x", "sigma_x", float(row["sigma_x"]), -4.0, atol=1.0e-10),
            _near("axisymmetric.sigma_z", "sigma_z", float(row["sigma_z"]), -4.0, atol=1.0e-10),
        ]

    return BenchmarkCase("thin_axisymmetric_k0", "axisymmetric", "Thin axisymmetric K0 geostatic reference", cfg, validate, tags=("axisymmetric", "k0"))


def _srm_case() -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "profile": PUBLIC_PROFILE},
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "nx": 1,
            "ny": 1,
            "element_type": "QUAD4",
            "material": "soil",
        },
        "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 80.0}},
        "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
        "loads": [{"edge": ["3", "4"], "ty": -10.0}],
        "steps": [{"name": "srm-reference", "type": "srm", "geofeas_workflow": "srm_slope", "srm": {"factors": [1.0, 2.0], "failure_plastic_ratio": 0.0}}],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        stage = result.stages[0]
        srm = stage.solver_info.get("srm", {})
        return [
            _near("srm.factor_of_safety", "factor_of_safety", float(srm.get("factor_of_safety", math.nan)), 2.0, atol=1.0e-12),
            _near("srm.trial_count", "trial_count", float(len(srm.get("trials", []))), 2.0, atol=0.0),
            _exists("srm.integration_point_history", stage.output_dir / "integration_point_stress.csv"),
        ]

    return BenchmarkCase("srm_strength_reduction", "srm", "SRM safety factor and integration-point history", cfg, validate, tags=("srm", "safety_factor"))


def _excavation_case() -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "profile": PUBLIC_PROFILE},
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 2.0],
            "y_range": [0.0, 1.0],
            "nx": 2,
            "ny": 1,
            "element_type": "QUAD4",
            "material": "soil",
        },
        "sets": {"elements": {"right": ["2"]}},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "boundary_conditions": [{"set": "left", "ux": 0.0, "uy": 0.0}, {"set": "bottom", "uy": 0.0}],
        "steps": [
            {"name": "before-excavation", "type": "static", "geofeas_workflow": "retaining_excavation", "stress_release": 0.4, "loads": [{"edge": ["4", "5"], "ty": -5.0}]},
            {"name": "excavate-right", "type": "excavation", "geofeas_workflow": "retaining_excavation", "stress_release": 0.6, "set": "right", "loads": [{"edge": ["4", "5"], "ty": -5.0}]},
        ],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        before, after = result.stages
        inactive = next(row for row in after.element_results if row["element_id"] == "2")
        return [
            _near("excavation.stage_count", "stage_count", float(len(result.stages)), 2.0, atol=0.0),
            _near("excavation.before_active", "active_element_count", float(len(before.active_elements)), 2.0, atol=0.0),
            _near("excavation.after_active", "active_element_count", float(len(after.active_elements)), 1.0, atol=0.0),
            _near("excavation.inactive_sigma_y", "inactive_sigma_y", float(inactive["sigma_y"]), 0.0, atol=1.0e-12),
        ]

    return BenchmarkCase("excavation_death", "stage", "Excavation/death stage with inactive element checks", cfg, validate, tags=("excavation", "death"))


def _liquefaction_case() -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"], "profile": PUBLIC_PROFILE},
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "nx": 1,
            "ny": 1,
            "element_type": "QUAD4",
            "material": "sand",
        },
        "materials": {
            "sand": {
                "model": "bilinear_liquefaction",
                "E": 10000.0,
                "nu": 0.30,
                "gamma": 18.0,
                "friction_angle": 32.0,
                "G0": 6000.0,
                "gamma_ref": 0.001,
                "liquefaction": {
                    "cyclic_stress_method": "gauss_overburden",
                    "initial_effective_stress": 5.0,
                    "cyclic_resistance_ratio": 0.2,
                    "cyclic_stress_ratio": 0.18,
                    "generation_rate": 0.2,
                    "dissipation_rate": 0.1,
                    "cycles_per_step": 1.0,
                },
            }
        },
        "boundary_conditions": [{"set": "all", "fixed": True}],
        "steps": [{"name": "liq-up", "type": "consolidation", "geofeas_workflow": "river_liquefaction_h28", "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0}}],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        stage = result.stages[0]
        liq = stage.solver_info.get("liquefaction", {})
        coupling = stage.solver_info.get("consolidation", {}).get("liquefaction_coupling", {})
        return [
            _gt("liquefaction.max_ru", "max_ru", float(liq.get("max_ru", 0.0)), 0.0),
            _gt("liquefaction.generation_source", "generation_source", float(coupling.get("generation_source", 0.0)), 0.0),
            _exists("liquefaction.state_csv", stage.output_dir / "liquefaction_state.csv"),
            _exists("liquefaction.history_csv", stage.output_dir / "liquefaction_history.csv"),
            _exists("liquefaction.ru_fl_svg", stage.output_dir / "liquefaction_ru_fl.svg"),
        ]

    return BenchmarkCase("liquefaction_up_history", "liquefaction", "u-p liquefaction ru update and history output", cfg, validate, tags=("liquefaction", "u-p"))


def _seepage_pressure_case(seepage_csv: Path) -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "profile": PUBLIC_PROFILE},
        "mesh": {
            "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
            "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            "node_sets": {"top": ["3", "4"]},
        },
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "ux": 0.0, "uy": 0.0}],
        "steps": [
            {
                "name": "external-seepage",
                "geofeas_workflow": "seepage_pressure",
                "time": 1.0,
                "hydro": {"initial_pressure": 0.0, "seepage_csv": str(seepage_csv), "water_levels": [{"set": "top", "water_level": 2.0}]},
            }
        ],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        stage = result.stages[0]
        pressure = stage.pore_pressure
        p0 = float(pressure[0]) if pressure is not None else math.nan
        p2 = float(pressure[2]) if pressure is not None else math.nan
        return [
            _near("seepage.node1_pressure", "pore_pressure", p0, 9.80665 * 2.0, atol=1.0e-8),
            _near("seepage.node3_pressure", "pore_pressure", p2, 9.80665 * 1.0, atol=1.0e-8),
            _near("seepage.sync_count", "seepage_sync_count", float(stage.solver_info.get("hydro_sync", {}).get("seepage_sync_count", 0.0)), 2.0, atol=0.0),
            _exists("seepage.pore_pressure_csv", stage.output_dir / "pore_pressure.csv"),
        ]

    return BenchmarkCase("seepage_water_pressure", "hydro", "External seepage head conversion and water pressure update", cfg, validate, tags=("seepage", "water_pressure"))


def _mpc_case() -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "profile": PUBLIC_PROFILE},
        "mesh": {
            "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
            "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
        },
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
        "loads": [{"node": "3", "fy": -10.0}],
        "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        return [
            _near("mpc.max_violation", "mpc_violation", float(stage.solver_info.get("mpc", {}).get("max_violation", math.inf)), 0.0, atol=1.0e-12),
            _near("mpc.tied_uy", "uy_difference", float(stage.displacements[2 * idx2 + 1] - stage.displacements[2 * idx3 + 1]), 0.0, atol=1.0e-12),
            _eq("mpc.method", "method", stage.solver_info.get("method"), "mpc_lagrange"),
        ]

    return BenchmarkCase("lagrange_mpc", "constraint", "Lagrange multiplier MPC displacement tie", cfg, validate, tags=("mpc", "lagrange"))


def _joint_case() -> BenchmarkCase:
    cfg = {
        "analysis": {"dimension": "2D", "type": "static_plane_strain", "profile": PUBLIC_PROFILE},
        "mesh": {"nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [0.0, 0.0], "4": [0.0, 1.0]}, "elements": []},
        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        "interfaces": [{"id": "joint", "minus_nodes": ["1", "2"], "plus_nodes": ["3", "4"], "kn": 1000.0, "kt": 1000.0, "behavior": {"friction": 0.5, "cohesion": 0.0}}],
        "boundary_conditions": [{"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0}, {"nodes": ["3", "4"], "ux": -0.01, "uy": 0.02}],
        "post": {"geofeas_style": True},
    }

    def validate(result: SolveResult2D) -> list[BenchmarkCheck]:
        stage = result.stages[0]
        plus_reaction_y = float(stage.reactions[5] + stage.reactions[7])
        slip = max(float(row.get("slip_abs", 0.0)) for row in stage.interface_results)
        states = {str(row.get("state", "")) for row in stage.interface_results}
        return [
            _near("joint.friction_reaction", "plus_reaction_y_abs", abs(plus_reaction_y), 5.0, atol=1.0e-10),
            _gt("joint.slip_abs", "slip_abs", slip, 0.0),
            _eq("joint.has_slip_state", "state", "slip" in states, True),
            _exists("joint.interface_state_csv", stage.output_dir / "interface_state.csv"),
        ]

    return BenchmarkCase("joint_mohr_coulomb", "interface", "Mohr-Coulomb joint slip and reaction post output", cfg, validate, tags=("joint", "interface"))


def _benchmark_metrics(result: SolveResult2D) -> dict[str, Any]:
    final = result.stages[-1]
    metrics: dict[str, Any] = {
        "node_count": len(result.mesh.node_ids),
        "element_count": len(result.mesh.elements),
        "stage_count": len(result.stages),
        "final_stage": final.name,
        "active_element_count": len(final.active_elements),
        "max_displacement": _max_displacement(result, final),
        "max_settlement": float(max((-final.displacements[1::2]), default=0.0)),
        "max_pore_pressure": None if final.pore_pressure is None else float(np.max(final.pore_pressure)),
        "solver_method": final.solver_info.get("method", ""),
    }
    if isinstance(final.solver_info.get("srm"), Mapping):
        metrics["srm_factor_of_safety"] = final.solver_info["srm"].get("factor_of_safety")
    if isinstance(final.solver_info.get("mpc"), Mapping):
        metrics["mpc_max_violation"] = final.solver_info["mpc"].get("max_violation")
    if final.interface_results:
        metrics["interface_slip_max"] = max(float(row.get("slip_abs", 0.0)) for row in final.interface_results)
    if isinstance(final.solver_info.get("liquefaction"), Mapping):
        metrics["liquefaction_max_ru"] = final.solver_info["liquefaction"].get("max_ru")
    return metrics


def _report_output_checks(result: SolveResult2D, required_outputs: tuple[str, ...]) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    for name in required_outputs:
        path = result.output_dir / name
        checks.append(_exists(f"report.{name}", path))
    pdf = result.output_dir / "calculation_report.pdf"
    if pdf.exists():
        checks.append(_eq("report.pdf_header", "pdf_header", pdf.read_bytes()[:5], b"%PDF-"))
    manifest_path = result.output_dir / "calculation_report_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks.append(_eq("report.manifest_frozen", "manifest_frozen", manifest.get("reproducibility", {}).get("frozen"), True))
            checks.append(_eq("report.manifest_input_hash_length", "input_sha256_length", len(str(manifest.get("reproducibility", {}).get("input_sha256", ""))), 64))
        except json.JSONDecodeError as exc:
            checks.append(BenchmarkCheck("report.manifest_json", "manifest_json", expected=True, actual=False, passed=False, detail=str(exc)))
    return checks


def _post_appearance_summary(result: SolveResult2D) -> dict[str, Any]:
    html_path = result.output_dir / "calculation_report.html"
    text = html_path.read_text(encoding="utf-8", errors="replace") if html_path.exists() else ""
    svg_count = len(re.findall(r"<svg\b", text))
    polygon_count = len(re.findall(r"<polygon\b", text))
    figure_caption_count = len(re.findall(r"<figcaption\b", text))
    color_count = len(set(re.findall(r"#[0-9a-fA-F]{6}", text)))
    legend_count = len(re.findall(r"class=\"legend\"", text))
    stage_reports = sum(1 for stage in result.stages if stage.output_dir and (stage.output_dir / "report.html").exists())
    liquefaction_figures = sum(1 for stage in result.stages if stage.output_dir and (stage.output_dir / "liquefaction_ru_fl.svg").exists())
    return {
        "html": str(html_path),
        "svg_count": svg_count,
        "polygon_count": polygon_count,
        "figure_caption_count": figure_caption_count,
        "color_count": color_count,
        "legend_count": legend_count,
        "stage_report_count": stage_reports,
        "liquefaction_figure_count": liquefaction_figures,
    }


def _post_appearance_checks(result: SolveResult2D, appearance: Mapping[str, Any]) -> list[BenchmarkCheck]:
    expected_svg = max(1, len(result.stages))
    checks = [
        _gte("post.svg_count", "svg_count", float(appearance.get("svg_count", 0)), float(expected_svg)),
        _gte("post.figure_captions", "figure_caption_count", float(appearance.get("figure_caption_count", 0)), float(expected_svg)),
        _gte("post.stage_reports", "stage_report_count", float(appearance.get("stage_report_count", 0)), float(len(result.stages))),
    ]
    if len(result.mesh.elements) > 0:
        checks.append(_gte("post.polygons", "polygon_count", float(appearance.get("polygon_count", 0)), float(len(result.mesh.elements))))
        checks.append(_gte("post.colors", "color_count", float(appearance.get("color_count", 0)), 3.0))
    return checks


def _max_displacement(result: SolveResult2D, stage: StageResult2D) -> float:
    return float(max((math.hypot(stage.displacements[2 * i], stage.displacements[2 * i + 1]) for i in range(len(result.mesh.node_ids))), default=0.0))


def _near(name: str, metric: str, actual: float, expected: float, *, rtol: float = 1.0e-5, atol: float = 1.0e-8) -> BenchmarkCheck:
    passed = math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected)
    return BenchmarkCheck(name, metric, expected, actual, rtol, atol, passed, f"abs_error={abs(actual - expected):.6g}")


def _gt(name: str, metric: str, actual: float, threshold: float) -> BenchmarkCheck:
    passed = math.isfinite(actual) and actual > threshold
    return BenchmarkCheck(name, metric, threshold, actual, 0.0, 0.0, passed, "actual > expected threshold")


def _gte(name: str, metric: str, actual: float, threshold: float) -> BenchmarkCheck:
    passed = math.isfinite(actual) and actual >= threshold
    return BenchmarkCheck(name, metric, threshold, actual, 0.0, 0.0, passed, "actual >= expected threshold")


def _eq(name: str, metric: str, actual: Any, expected: Any) -> BenchmarkCheck:
    return BenchmarkCheck(name, metric, expected, actual, 0.0, 0.0, actual == expected, "actual == expected")


def _exists(name: str, path: Path | None) -> BenchmarkCheck:
    exists = path is not None and Path(path).exists()
    return BenchmarkCheck(name, "file_exists", True, exists, 0.0, 0.0, exists, str(path))


def _check_to_row(case: BenchmarkCase, check: BenchmarkCheck) -> dict[str, Any]:
    return {
        "case": case.name,
        "category": case.category,
        "check": check.name,
        "metric": check.metric,
        "expected": check.expected,
        "actual": check.actual,
        "rtol": check.rtol,
        "atol": check.atol,
        "passed": check.passed,
        "detail": check.detail,
    }


def _write_case_tolerance_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "category", "check", "metric", "expected", "actual", "rtol", "atol", "passed", "detail"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_metrics_csv(cases: list[Mapping[str, Any]], path: Path) -> None:
    keys = sorted({key for case in cases for key in dict(case.get("metrics", {})).keys()})
    fields = ["case", "category", "passed", "elapsed_seconds", *keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            metrics = dict(case.get("metrics", {}))
            writer.writerow({"case": case.get("name", ""), "category": case.get("category", ""), "passed": case.get("passed", ""), "elapsed_seconds": case.get("elapsed_seconds", ""), **metrics})


def _write_benchmark_html(summary: Mapping[str, Any], checks: list[Mapping[str, Any]], path: Path) -> None:
    case_rows = []
    for case in summary.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        case_rows.append(
            "<tr class='{cls}'><td>{name}</td><td>{category}</td><td>{passed}</td><td>{failed}</td><td>{elapsed:.4f}</td><td>{out}</td></tr>".format(
                cls="ok" if bool(case.get("passed", False)) else "ng",
                name=html.escape(str(case.get("name", ""))),
                category=html.escape(str(case.get("category", ""))),
                passed="OK" if bool(case.get("passed", False)) else "NG",
                failed=int(case.get("failed_count", 0) or 0),
                elapsed=float(case.get("elapsed_seconds", 0.0) or 0.0),
                out=html.escape(str(case.get("output_dir", ""))),
            )
        )
    check_rows = []
    for row in checks:
        ok = bool(row.get("passed", False))
        check_rows.append(
            "<tr class='{cls}'><td>{case}</td><td>{check}</td><td>{metric}</td><td>{expected}</td><td>{actual}</td><td>{tol}</td><td>{passed}</td></tr>".format(
                cls="ok" if ok else "ng",
                case=html.escape(str(row.get("case", ""))),
                check=html.escape(str(row.get("check", ""))),
                metric=html.escape(str(row.get("metric", ""))),
                expected=html.escape(str(row.get("expected", ""))),
                actual=html.escape(str(row.get("actual", ""))),
                tol=html.escape(f"rtol={row.get('rtol', '')}, atol={row.get('atol', '')}"),
                passed="OK" if ok else "NG",
            )
        )
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS-like benchmark report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.ng {{ background: #fef2f2; }}
.status {{ font-weight: 700; }}
</style>
</head>
<body>
<h1>GeoFEAS-like benchmark report</h1>
<p class="status">Status: {'PASSED' if bool(summary.get('passed', False)) else 'FAILED'}</p>
<p>case_count={int(summary.get('case_count', 0) or 0)}, check_count={int(summary.get('check_count', 0) or 0)}, failed_count={int(summary.get('failed_count', 0) or 0)}, elapsed={float(summary.get('elapsed_seconds', 0.0) or 0.0):.4f}s</p>
<h2>Cases</h2>
<table><thead><tr><th>case</th><th>category</th><th>judgement</th><th>failed</th><th>elapsed</th><th>output</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table>
<h2>Tolerance checks</h2>
<table><thead><tr><th>case</th><th>check</th><th>metric</th><th>expected</th><th>actual</th><th>tolerance</th><th>judgement</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


__all__ = [
    "BenchmarkCase",
    "BenchmarkCheck",
    "build_geofeas_benchmark_cases",
    "run_geofeas_benchmark_suite",
]
