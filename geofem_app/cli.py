"""Command line entry point for GeoFEM."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import yaml

from .case_management import write_case_manifest, write_failure_report
from .api_contracts import write_api_contract_docs
from .commercial_quality import run_commercial_quality_check
from .customization import apply_organization_profile, default_organization_profile, load_organization_profile, validate_organization_profile, write_customization_artifacts
from .encoding_policy import configure_utf8_console, write_encoding_audit
from .fem2d import FEM2DError, mesh_from_config, solve_plane_strain_config
from .input_diagnostics import diagnose_input_config, write_input_diagnostics
from .maintainability_audit import run_maintainability_audit
from .material_models import write_material_reports
from .mesh_quality import apply_repair_candidates_to_config, evaluate_mesh_quality, write_mesh_quality_report
from .output_location import resolve_analysis_output_dir, write_run_manifest
from .output_comparison import compare_result_cases
from .project import ensure_project_dirs, new_default_project, save_project, update_after_run
from .sample_projects import sample_project_catalog, write_sample_project_suite
from .samples import plane_strain_patch_sample, plane_strain_quad4_sample
from .standard_benchmarks import run_standard_benchmark_suite
from .startup_check import run_startup_check
from .startup_support import DEFAULT_SUPPORT_EXCLUDE_PATTERNS, MAX_SUPPORT_FILE_BYTES, SupportPackageOptions
from .update_compatibility import write_update_compatibility_artifacts
from .vgflow2d import is_vgflow2d_config, solve_vgflow2d_config
from .workspace_management import write_workspace_dashboard


def main(argv: list[str] | None = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(prog="geofem", description="GeoFEM 2D plane-strain FEM application")
    sub = parser.add_subparsers(dest="command")

    p_solve = sub.add_parser("solve", help="run a 2D GeoFEM input")
    p_solve.add_argument("input", help="YAML/JSON input file")
    p_solve.add_argument("--out", "--output-dir", dest="output_dir", help="output directory")
    p_solve.add_argument("--srm-workers", help="SRM parallel worker count or 'auto' for CLI/batch runs")
    p_solve.add_argument("--srm-parallel-policy", choices=["interactive", "batch", "auto"], help="SRM worker policy for this solve")
    p_solve.add_argument("--srm-memory-limit-mb", type=float, help="memory budget used to cap SRM worker count")
    p_solve.add_argument("--srm-memory-per-worker-mb", type=float, help="estimated incremental memory per SRM worker")
    p_solve.add_argument("--cancel-file", help="path whose creation requests cooperative cancellation during long SRM lookahead runs")

    p_diag = sub.add_parser("diagnose", help="validate and diagnose a GeoFEM input without solving")
    p_diag.add_argument("input", help="YAML/JSON input file")
    p_diag.add_argument("--out", "--output-dir", dest="output_dir", help="diagnostic output directory")
    p_diag.add_argument("--fail-on-warning", action="store_true")

    p_upgrade = sub.add_parser("upgrade-check", help="check input schema migration and result artifact revalidation")
    p_upgrade.add_argument("input", help="YAML/JSON input file")
    p_upgrade.add_argument("--out", "--output-dir", dest="output_dir", help="update compatibility output directory")
    p_upgrade.add_argument("--previous-version", help="previous GeoFEM version recorded for the project")
    p_upgrade.add_argument("--artifacts", help="existing result artifact directory to revalidate")
    p_upgrade.add_argument("--fail-on-warning", action="store_true", help="fail if migration or artifact warnings are reported")

    p_quality = sub.add_parser("mesh-quality", help="evaluate mesh quality and optionally write repair-ready input")
    p_quality.add_argument("input", help="YAML/JSON input file")
    p_quality.add_argument("--out", "--output-dir", dest="output_dir", help="mesh-quality output directory")
    p_quality.add_argument("--write-repaired", help="write an input file with repair candidates applied as mesh controls")

    p_materials = sub.add_parser("materials", help="write material-model catalog and material inventory")
    p_materials.add_argument("input", nargs="?", help="optional YAML/JSON input file")
    p_materials.add_argument("--out", "--output-dir", dest="output_dir", default="runs/material_models", help="material report output directory")

    p_bench = sub.add_parser("benchmarks", help="run GeoFEM-native standard benchmarks")
    p_bench.add_argument("--out", default="runs/geofem_standard_benchmarks", help="output directory")
    p_bench.add_argument("--baseline", help="baseline standard_benchmark_performance.json to compare against")
    p_bench.add_argument("--max-slowdown", type=float, default=1.25, help="allowed slowdown ratio for performance regression checks")
    p_bench.add_argument("--max-structure-growth", type=float, help="allowed max-matrix-nnz growth ratio (defaults to --max-slowdown)")
    p_bench.add_argument("--max-memory-growth", type=float, help="allowed estimated-memory growth ratio (defaults to --max-slowdown)")
    p_bench.add_argument("--max-fallback-increase", type=int, default=0, help="allowed fallback count increase")
    p_bench.add_argument("--update-baseline", action="store_true", help="write a new performance baseline beside benchmark outputs")
    p_bench.add_argument("--skip-numba-warmup", action="store_true", help="skip representative Numba warmup before benchmark case timing")

    p_warmup = sub.add_parser("warmup", help="compile representative Numba kernels and exit")
    p_warmup.add_argument("--profile", default="gui", help="warmup kernel profile")

    p_doctor = sub.add_parser("doctor", help="verify dependencies, helper files, and a first sample solve")
    p_doctor.add_argument("--out", default="runs/startup_check", help="startup-check output directory")
    p_doctor.add_argument("--include-gui", action="store_true", help="treat GUI dependencies as required")
    p_doctor.add_argument("--skip-sample", action="store_true", help="skip the sample solve")
    p_doctor.add_argument("--support-include-personal-info", action="store_true", help="do not redact personal paths from the support ZIP")
    p_doctor.add_argument("--support-include-large-results", action="store_true", help="include files above the support ZIP size limit")
    p_doctor.add_argument("--support-max-file-bytes", type=int, default=MAX_SUPPORT_FILE_BYTES, help="maximum file size included in the support ZIP")
    p_doctor.add_argument("--support-exclude-pattern", action="append", default=[], help="additional support ZIP exclude glob pattern")

    p_contracts = sub.add_parser("api-contracts", help="write documented internal API contract artifacts")
    p_contracts.add_argument("--out", default="runs/api_contracts", help="contract documentation output directory")

    p_maint = sub.add_parser("maintainability-audit", help="scan large files and mixed responsibility boundaries")
    p_maint.add_argument("--out", default="runs/maintainability_audit", help="audit output directory")
    p_maint.add_argument("--threshold", type=int, default=1000, help="line count threshold for large-file candidates")

    p_commercial = sub.add_parser("commercial-quality", help="run commercial-grade result, report, and performance quality gates")
    p_commercial.add_argument("--out", default="runs/commercial_quality", help="quality-gate output directory")
    p_commercial.add_argument("--result", help="optional completed result directory to validate")
    p_commercial.add_argument("--baseline", help="optional Post visual baseline directory")
    p_commercial.add_argument("--benchmark-baseline", help="optional standard_benchmark_performance.json baseline")
    p_commercial.add_argument("--max-slowdown", type=float, default=1.25, help="allowed slowdown ratio for benchmark performance checks")
    p_commercial.add_argument("--skip-benchmarks", action="store_true", help="skip standard benchmark and performance gates")
    p_commercial.add_argument("--include-startup", action="store_true", help="include startup/environment checks")
    p_commercial.add_argument("--include-gui", action="store_true", help="treat GUI dependencies as required during startup checks")
    p_commercial.add_argument("--fail-on-warning", action="store_true", help="fail if optional warning-level modules report warnings")

    p_compare = sub.add_parser("compare-results", help="compare two completed GeoFEM result directories")
    p_compare.add_argument("current", help="current/actual result directory or run directory")
    p_compare.add_argument("baseline", help="previous, baseline, or design-case result directory")
    p_compare.add_argument("--out", default="runs/case_output_comparison", help="comparison output directory")
    p_compare.add_argument("--current-label", default="current")
    p_compare.add_argument("--baseline-label", default="baseline")
    p_compare.add_argument("--atol", type=float, default=1.0e-9)
    p_compare.add_argument("--rtol", type=float, default=1.0e-6)

    p_encoding = sub.add_parser("encoding-audit", help="audit source/docs/artifacts for UTF-8 and mojibake risks")
    p_encoding.add_argument("--root", default=".", help="root file or directory to audit")
    p_encoding.add_argument("--out", default="runs/encoding_audit", help="encoding audit output directory")
    p_encoding.add_argument("--fail-on-warning", action="store_true", help="treat warning-level encoding findings as failures")

    p_init = sub.add_parser("init", help="create a project folder")
    p_init.add_argument("directory", help="project directory")
    p_init.add_argument("--name", default="GeoFEM Project")
    p_init.add_argument("--sample", action="store_true", help="write a runnable sample input")

    p_sample = sub.add_parser("sample", help="write a built-in sample input")
    p_sample.add_argument("path", help="output YAML path")
    p_sample.add_argument("--kind", choices=["quad4", "patch"], default="quad4")
    p_sample.add_argument("--element", default="QUAD4")
    p_sample.add_argument("--integration", default="B-bar")

    p_sample_projects = sub.add_parser("sample-projects", help="write practical tutorial project folders")
    p_sample_projects.add_argument("--out", default="examples/sample_projects", help="sample project suite output directory")
    p_sample_projects.add_argument("--case", action="append", help="case id to write; repeat or use all")

    p_workspace = sub.add_parser("workspace-dashboard", help="write project dashboard, artifact inventory, storage summary, and optional archive")
    p_workspace.add_argument("root", nargs="?", default=".", help="project workspace root")
    p_workspace.add_argument("--out", default="runs/workspace_dashboard", help="workspace dashboard output directory")
    p_workspace.add_argument("--archive", action="store_true", help="also write a portable workspace ZIP archive")
    p_workspace.add_argument("--storage-warning-mb", type=float, default=1024.0, help="storage warning threshold in MB")

    p_customize = sub.add_parser("customization", help="write and apply organization customization profiles")
    p_customize.add_argument("--out", default="runs/customization", help="customization artifact output directory")
    p_customize.add_argument("--profile", help="organization profile JSON/YAML to use")
    p_customize.add_argument("--input", help="optional input YAML/JSON to customize")
    p_customize.add_argument("--template", help="project template id to apply")
    p_customize.add_argument("--applied-output", help="path for customized input YAML when --input is provided")

    sub.add_parser("gui", help="start the PySide6 GUI")

    if argv is None and len(sys.argv) == 2 and Path(sys.argv[1]).suffix.lower() in {".yaml", ".yml", ".json"}:
        argv = ["solve", sys.argv[1]]
    args = parser.parse_args(argv)

    try:
        if args.command == "solve":
            return run_solve(
                Path(args.input),
                output_dir=args.output_dir,
                srm_workers=args.srm_workers,
                srm_parallel_policy=args.srm_parallel_policy,
                srm_memory_limit_mb=args.srm_memory_limit_mb,
                srm_memory_per_worker_mb=args.srm_memory_per_worker_mb,
                cancel_file=args.cancel_file,
            )
        if args.command == "diagnose":
            return run_diagnose(Path(args.input), output_dir=args.output_dir, fail_on_warning=args.fail_on_warning)
        if args.command == "upgrade-check":
            return run_upgrade_check(
                Path(args.input),
                output_dir=args.output_dir,
                previous_version=args.previous_version,
                artifact_dir=args.artifacts,
                fail_on_warning=args.fail_on_warning,
            )
        if args.command == "mesh-quality":
            return run_mesh_quality(Path(args.input), output_dir=args.output_dir, write_repaired=args.write_repaired)
        if args.command == "materials":
            return run_materials(Path(args.input) if args.input else None, output_dir=args.output_dir)
        if args.command == "benchmarks":
            return run_benchmarks(
                Path(args.out),
                baseline=args.baseline,
                max_slowdown=args.max_slowdown,
                max_structure_growth=args.max_structure_growth,
                max_memory_growth=args.max_memory_growth,
                max_fallback_increase=args.max_fallback_increase,
                update_baseline=args.update_baseline,
                numba_warmup=not args.skip_numba_warmup,
            )
        if args.command == "warmup":
            return run_numba_warmup(profile=args.profile)
        if args.command == "doctor":
            return run_doctor(
                Path(args.out),
                include_gui=args.include_gui,
                run_sample=not args.skip_sample,
                support_include_personal_info=args.support_include_personal_info,
                support_include_large_results=args.support_include_large_results,
                support_max_file_bytes=args.support_max_file_bytes,
                support_exclude_patterns=args.support_exclude_pattern,
            )
        if args.command == "api-contracts":
            return run_api_contracts(Path(args.out))
        if args.command == "maintainability-audit":
            return run_maintainability(Path(args.out), threshold=args.threshold)
        if args.command == "commercial-quality":
            return run_commercial_quality(
                Path(args.out),
                result_dir=args.result,
                baseline_dir=args.baseline,
                run_benchmarks=not args.skip_benchmarks,
                benchmark_baseline=args.benchmark_baseline,
                max_slowdown=args.max_slowdown,
                include_startup=args.include_startup,
                include_gui=args.include_gui,
                fail_on_warning=args.fail_on_warning,
            )
        if args.command == "compare-results":
            return run_compare_results(
                Path(args.current),
                Path(args.baseline),
                output_dir=Path(args.out),
                current_label=args.current_label,
                baseline_label=args.baseline_label,
                abs_tolerance=args.atol,
                rel_tolerance=args.rtol,
            )
        if args.command == "encoding-audit":
            return run_encoding_audit(Path(args.root), Path(args.out), fail_on_warning=args.fail_on_warning)
        if args.command == "init":
            return run_init(Path(args.directory), name=args.name, sample=args.sample)
        if args.command == "sample":
            return write_sample(Path(args.path), kind=args.kind, element=args.element, integration=args.integration)
        if args.command == "sample-projects":
            return run_sample_projects(Path(args.out), cases=args.case)
        if args.command == "workspace-dashboard":
            return run_workspace_dashboard(
                Path(args.root),
                output_dir=Path(args.out),
                create_archive=args.archive,
                storage_warning_bytes=int(max(0.0, args.storage_warning_mb) * 1_000_000),
            )
        if args.command == "customization":
            return run_customization(
                Path(args.out),
                profile_path=Path(args.profile) if args.profile else None,
                input_path=Path(args.input) if args.input else None,
                template_id=args.template,
                applied_output=Path(args.applied_output) if args.applied_output else None,
            )
        if args.command == "gui":
            from .gui.main_window import run_gui

            return run_gui()
        parser.print_help()
        return 2
    except FEM2DError as exc:
        print(f"[GeoFEM][input error] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[GeoFEM][error] {exc}", file=sys.stderr)
        return 1


def run_solve(
    input_path: Path,
    *,
    output_dir: str | None = None,
    srm_workers: str | int | None = None,
    srm_parallel_policy: str | None = None,
    srm_memory_limit_mb: float | None = None,
    srm_memory_per_worker_mb: float | None = None,
    cancel_file: str | None = None,
) -> int:
    out = Path(output_dir or _default_run_dir(input_path)).resolve()
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    diagnostics: dict[str, Any] | None = None
    output_cfg: Mapping[str, Any] = {}
    try:
        cfg = load_input(input_path)
        cfg = _apply_cli_solver_runtime_defaults(
            cfg,
            srm_workers=srm_workers,
            srm_parallel_policy=srm_parallel_policy,
            srm_memory_limit_mb=srm_memory_limit_mb,
            srm_memory_per_worker_mb=srm_memory_per_worker_mb,
            cancel_file=cancel_file,
        )
        output_cfg = cfg.get("output", cfg.get("outputs", {})) if isinstance(cfg, Mapping) else {}
        if not isinstance(output_cfg, Mapping):
            output_cfg = {}
        out = resolve_analysis_output_dir(
            input_path,
            output_cfg,
            Path.cwd(),
            explicit_out=output_dir,
            default_root_policy="project_runs",
        )
        diagnostics = diagnose_input_config(cfg)
        write_input_diagnostics(diagnostics, out)
        dim = detect_dimension(cfg)
        if dim != "2D":
            raise FEM2DError(f"3D analysis has been removed from this application; only 2D plane-strain inputs are supported (detected {dim})")
        if int(diagnostics.get("error_count", 0) or 0) > 0:
            first = next((issue for issue in diagnostics.get("issues", []) if isinstance(issue, Mapping) and issue.get("severity") == "ERROR"), {})
            raise FEM2DError(f"input diagnostics failed: {first.get('path', 'input')}: {first.get('message', 'invalid input')}")
        if is_vgflow2d_config(cfg):
            vg_result = solve_vgflow2d_config(cfg, out)
            elapsed = time.perf_counter() - t0
            write_case_manifest(
                output_dir=vg_result.output_dir,
                input_path=input_path,
                diagnostics=diagnostics,
                status="completed",
                started_at=started,
                elapsed_seconds=elapsed,
                result_summary={"stage_count": len(vg_result.steps), "warning_count": len(vg_result.warnings), "analysis": "vgflow2d"},
            )
            write_run_manifest(
                output_dir=vg_result.output_dir,
                input_path=input_path,
                status="completed",
                started_at=started,
                elapsed_seconds=elapsed,
                output_config=output_cfg,
                extra={"analysis": "vgflow2d", "stage_count": len(vg_result.steps), "warning_count": len(vg_result.warnings)},
            )
            print(f"[GeoFEM VGFlow2D] completed: {vg_result.output_dir}")
            for warning in vg_result.warnings:
                print(f"[GeoFEM VGFlow2D][warning] {warning}")
            return 0

        result = solve_plane_strain_config(cfg, out)
        elapsed = time.perf_counter() - t0
        write_case_manifest(
            output_dir=result.output_dir,
            input_path=input_path,
            diagnostics=diagnostics,
            status="completed",
            started_at=started,
            elapsed_seconds=elapsed,
            result_summary={"stage_count": len(result.stages), "warning_count": len(result.warnings)},
        )
        write_run_manifest(
            output_dir=result.output_dir,
            input_path=input_path,
            status="completed",
            started_at=started,
            elapsed_seconds=elapsed,
            output_config=output_cfg,
            extra={"analysis": "2d", "stage_count": len(result.stages), "warning_count": len(result.warnings)},
        )
        print(f"[GeoFEM 2D] completed: {result.output_dir}")
        for warning in result.warnings:
            print(f"[GeoFEM 2D][warning] {warning}")
        return 0
    except Exception as exc:
        try:
            if diagnostics is not None:
                write_input_diagnostics(diagnostics, out)
            write_failure_report(output_dir=out, input_path=input_path, error=exc, diagnostics=diagnostics, started_at=started)
            write_case_manifest(
                output_dir=out,
                input_path=input_path,
                diagnostics=diagnostics,
                status="failed",
                started_at=started,
                elapsed_seconds=time.perf_counter() - t0,
                error=exc,
            )
            write_run_manifest(
                output_dir=out,
                input_path=input_path,
                status="failed",
                started_at=started,
                elapsed_seconds=time.perf_counter() - t0,
                output_config=output_cfg,
                extra={"error": str(exc)},
            )
        except Exception:
            pass
        raise


def _apply_cli_solver_runtime_defaults(
    cfg: Any,
    *,
    srm_workers: str | int | None = None,
    srm_parallel_policy: str | None = None,
    srm_memory_limit_mb: float | None = None,
    srm_memory_per_worker_mb: float | None = None,
    cancel_file: str | None = None,
) -> Any:
    if not isinstance(cfg, Mapping):
        return cfg
    updated: dict[str, Any] = copy.deepcopy(dict(cfg))
    solver = updated.get("solver", {})
    solver_map: dict[str, Any] = dict(solver) if isinstance(solver, Mapping) else {}
    execution = solver_map.get("execution", {})
    execution_map: dict[str, Any] = dict(execution) if isinstance(execution, Mapping) else {}
    execution_map.setdefault("context", "cli")
    execution_map.setdefault("profile", "batch")
    if cancel_file:
        execution_map["cancel_file"] = str(Path(cancel_file).resolve())
    solver_map["execution"] = execution_map

    has_srm_override = any(value is not None for value in (srm_workers, srm_parallel_policy, srm_memory_limit_mb, srm_memory_per_worker_mb))
    if has_srm_override:
        srm = solver_map.get("srm", {})
        srm_map: dict[str, Any] = dict(srm) if isinstance(srm, Mapping) else {}
        parallel = srm_map.get("parallel", {})
        if isinstance(parallel, Mapping):
            parallel_map: dict[str, Any] = dict(parallel)
        else:
            parallel_map = {"enabled": bool(parallel)}
        parallel_map["enabled"] = True
        parallel_map["policy"] = str(srm_parallel_policy or parallel_map.get("policy", "batch"))
        if srm_workers is not None:
            parallel_map["max_workers"] = _cli_srm_workers_value(srm_workers)
        if srm_memory_limit_mb is not None:
            parallel_map["memory_limit_mb"] = float(srm_memory_limit_mb)
        if srm_memory_per_worker_mb is not None:
            parallel_map["memory_per_worker_mb"] = float(srm_memory_per_worker_mb)
        srm_map["parallel"] = parallel_map
        solver_map["srm"] = srm_map

    updated["solver"] = solver_map
    return updated


def _cli_srm_workers_value(value: str | int) -> str | int:
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        return max(1, int(text))
    except (TypeError, ValueError):
        return text


def run_diagnose(input_path: Path, *, output_dir: str | None = None, fail_on_warning: bool = False) -> int:
    out = Path(output_dir or (input_path.resolve().parent / "diagnostics" / input_path.stem))
    cfg = load_input(input_path)
    diagnostics = diagnose_input_config(cfg)
    artifacts = write_input_diagnostics(diagnostics, out)
    print(json.dumps({"passed": diagnostics["passed"], "error_count": diagnostics["error_count"], "warning_count": diagnostics["warning_count"], **artifacts}, ensure_ascii=False))
    if int(diagnostics["error_count"]) > 0:
        return 1
    if fail_on_warning and int(diagnostics["warning_count"]) > 0:
        return 1
    return 0


def run_upgrade_check(
    input_path: Path,
    *,
    output_dir: str | None = None,
    previous_version: str | None = None,
    artifact_dir: str | None = None,
    fail_on_warning: bool = False,
) -> int:
    out = Path(output_dir or (input_path.resolve().parent / "upgrade_check" / input_path.stem))
    cfg = load_input(input_path)
    paths = write_update_compatibility_artifacts(cfg, out, previous_version=previous_version, artifact_dir=artifact_dir)
    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "passed": bool(report.get("passed", False)),
                "error_count": int(report.get("error_count", 0) or 0),
                "warning_count": int(report.get("warning_count", 0) or 0),
                **paths,
            },
            ensure_ascii=False,
        )
    )
    if int(report.get("error_count", 0) or 0) > 0:
        return 1
    if fail_on_warning and int(report.get("warning_count", 0) or 0) > 0:
        return 1
    return 0


def run_mesh_quality(input_path: Path, *, output_dir: str | None = None, write_repaired: str | None = None) -> int:
    cfg = load_input(input_path)
    mesh = mesh_from_config(cfg)
    out = Path(output_dir or (input_path.resolve().parent / "mesh_quality" / input_path.stem))
    paths = write_mesh_quality_report(mesh, out, cfg)
    report = evaluate_mesh_quality(mesh, cfg)
    applied: list[str] = []
    if write_repaired:
        repaired, applied = apply_repair_candidates_to_config(cfg, report.get("repair_candidates", []))
        Path(write_repaired).parent.mkdir(parents=True, exist_ok=True)
        Path(write_repaired).write_text(yaml.safe_dump(repaired, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": bool(report.get("passed", False)),
                "violation_count": int(report.get("summary", {}).get("violation_count", 0) if isinstance(report.get("summary", {}), Mapping) else 0),
                "repair_candidate_count": len(report.get("repair_candidates", [])),
                "applied_repair_count": len(applied),
                **paths,
            },
            ensure_ascii=False,
        )
    )
    return 0 if bool(report.get("passed", False)) else 1


def run_materials(input_path: Path | None, *, output_dir: str) -> int:
    cfg: dict[str, Any] = {}
    materials: Mapping[str, Any] = {}
    if input_path is not None:
        cfg = load_input(input_path)
        materials = cfg.get("materials", cfg.get("material", {})) if isinstance(cfg.get("materials", cfg.get("material", {})), Mapping) else {}
    paths = write_material_reports(materials, output_dir, input_config=cfg)
    print(json.dumps({"out": str(output_dir), **paths}, ensure_ascii=False))
    return 0


def run_numba_warmup(*, profile: str = "gui") -> int:
    """Warm solver kernels in a short-lived process for GUI startup."""

    from .numba_warmup import warmup_numba_kernels

    summary = warmup_numba_kernels(profile=str(profile or "gui"))
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 1 if int(summary.get("failed_count", 0) or 0) else 0


def run_benchmarks(
    output_dir: Path,
    *,
    baseline: str | None = None,
    max_slowdown: float = 1.25,
    max_structure_growth: float | None = None,
    max_memory_growth: float | None = None,
    max_fallback_increase: int = 0,
    update_baseline: bool = False,
    numba_warmup: bool = True,
) -> int:
    summary = run_standard_benchmark_suite(
        output_dir,
        baseline=baseline,
        max_slowdown=max_slowdown,
        max_structure_growth=max_structure_growth,
        max_memory_growth=max_memory_growth,
        max_fallback_increase=max_fallback_increase,
        update_baseline=update_baseline,
        numba_warmup=numba_warmup,
    )
    print(
        json.dumps(
            {
                "passed": bool(summary.get("passed", False)),
                "case_count": int(summary.get("case_count", 0) or 0),
                "failed_count": int(summary.get("failed_count", 0) or 0),
                "performance_regression_count": int(summary.get("performance_regression_count", 0) or 0),
                "numba_warmup_elapsed_seconds": float(summary.get("numba_warmup_elapsed_seconds", 0.0) or 0.0),
                "case_elapsed_seconds_excluding_warmup": float(summary.get("case_elapsed_seconds_excluding_warmup", 0.0) or 0.0),
                "out": str(output_dir),
                "summary": str(output_dir / "standard_benchmark_summary.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary.get("passed", False) else 1


def run_doctor(
    output_dir: Path,
    *,
    include_gui: bool = False,
    run_sample: bool = True,
    support_include_personal_info: bool = False,
    support_include_large_results: bool = False,
    support_max_file_bytes: int = MAX_SUPPORT_FILE_BYTES,
    support_exclude_patterns: list[str] | tuple[str, ...] | None = None,
) -> int:
    support_options = SupportPackageOptions(
        exclude_personal_info=not support_include_personal_info,
        include_large_results=support_include_large_results,
        max_file_bytes=max(1, int(support_max_file_bytes or MAX_SUPPORT_FILE_BYTES)),
        exclude_patterns=tuple(DEFAULT_SUPPORT_EXCLUDE_PATTERNS) + tuple(support_exclude_patterns or ()),
    )
    report = run_startup_check(output_dir, include_gui=include_gui, run_sample=run_sample, support_options=support_options)
    print(
        json.dumps(
            {
                "passed": bool(report.get("passed", False)),
                "error_count": int(report.get("error_count", 0) or 0),
                "warning_count": int(report.get("warning_count", 0) or 0),
                "out": str(output_dir),
                "report": str(output_dir / "startup_check.json"),
                "support_manifest": str(output_dir / "startup_support_manifest.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.get("passed", False) else 1


def run_api_contracts(output_dir: Path) -> int:
    paths = write_api_contract_docs(output_dir)
    print(json.dumps({"out": str(output_dir), **paths}, ensure_ascii=False))
    return 0


def run_maintainability(output_dir: Path, *, threshold: int = 1000) -> int:
    root = Path(__file__).resolve().parents[1]
    summary = run_maintainability_audit(root, output_dir, large_line_threshold=threshold)
    print(
        json.dumps(
            {
                "candidate_count": int(summary.get("candidate_count", 0) or 0),
                "file_count": int(summary.get("file_count", 0) or 0),
                "out": str(output_dir),
                **dict(summary.get("paths", {})),
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_commercial_quality(
    output_dir: Path,
    *,
    result_dir: str | None = None,
    baseline_dir: str | None = None,
    run_benchmarks: bool = True,
    benchmark_baseline: str | None = None,
    max_slowdown: float = 1.25,
    include_startup: bool = False,
    include_gui: bool = False,
    fail_on_warning: bool = False,
) -> int:
    summary = run_commercial_quality_check(
        output_dir,
        result_dir=result_dir,
        baseline_dir=baseline_dir,
        run_benchmarks=run_benchmarks,
        benchmark_baseline=benchmark_baseline,
        max_slowdown=max_slowdown,
        include_startup=include_startup,
        include_gui=include_gui,
        fail_on_warning=fail_on_warning,
    )
    print(
        json.dumps(
            {
                "passed": bool(summary.get("passed", False)),
                "failed_required_count": int(summary.get("failed_required_count", 0) or 0),
                "warning_module_count": int(summary.get("warning_module_count", 0) or 0),
                "out": str(output_dir),
                "summary": str(output_dir / "commercial_quality_check.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary.get("passed", False) else 1


def run_compare_results(
    current: Path,
    baseline: Path,
    *,
    output_dir: Path,
    current_label: str = "current",
    baseline_label: str = "baseline",
    abs_tolerance: float = 1.0e-9,
    rel_tolerance: float = 1.0e-6,
) -> int:
    comparison = compare_result_cases(
        current,
        baseline,
        output_dir=output_dir,
        current_label=current_label,
        baseline_label=baseline_label,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    print(
        json.dumps(
            {
                "passed": bool(comparison.get("passed", False)),
                "difference_count": int(comparison.get("difference_count", 0) or 0),
                "missing_count": int(comparison.get("missing_count", 0) or 0),
                "row_count": int(comparison.get("row_count", 0) or 0),
                "out": str(output_dir),
                "summary": str(output_dir / "case_output_comparison.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if comparison.get("passed", False) else 1


def run_encoding_audit(root: Path, output_dir: Path, *, fail_on_warning: bool = False) -> int:
    summary = write_encoding_audit(root, output_dir, fail_on_warning=fail_on_warning)
    passed = bool(summary.get("passed_with_warning_policy", summary.get("passed", False)))
    print(
        json.dumps(
            {
                "passed": passed,
                "error_count": int(summary.get("error_count", 0) or 0),
                "warning_count": int(summary.get("warning_count", 0) or 0),
                "file_count": int(summary.get("file_count", 0) or 0),
                "out": str(output_dir),
                **dict(summary.get("paths", {})),
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


def run_init(directory: Path, *, name: str, sample: bool) -> int:
    ensure_project_dirs(directory)
    project = new_default_project(directory, name=name)
    project_path = save_project(project)
    print(f"[GeoFEM] project created: {project_path}")
    if sample:
        input_path = directory / "input" / "sample_2d.yaml"
        write_sample(input_path, kind="quad4", element="QUAD4", integration="B-bar")
        project.input_file = str(input_path)
        update_after_run(project, directory / "runs")
        save_project(project, project_path)
    return 0


def write_sample(path: Path, *, kind: str, element: str, integration: str) -> int:
    if kind == "patch":
        data = plane_strain_patch_sample(element, integration)
    else:
        data = plane_strain_quad4_sample(integration=integration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[GeoFEM] sample written: {path}")
    return 0


def run_sample_projects(output_dir: Path, *, cases: list[str] | None = None) -> int:
    paths = write_sample_project_suite(output_dir, cases=cases)
    catalog = sample_project_catalog()
    selected = "all" if not cases or any(str(case).lower() == "all" for case in cases) else ",".join(cases)
    print(json.dumps({"case_selection": selected, "case_count": len(catalog) if selected == "all" else len(cases or []), "out": str(output_dir), **paths}, ensure_ascii=False))
    return 0


def run_workspace_dashboard(
    root: Path,
    *,
    output_dir: Path,
    create_archive: bool = False,
    storage_warning_bytes: int = 1_000_000_000,
) -> int:
    paths = write_workspace_dashboard(
        root,
        output_dir,
        create_archive=create_archive,
        storage_warning_bytes=storage_warning_bytes,
    )
    payload = {
        "passed": True,
        "root": str(root),
        "out": str(output_dir),
        "archive_created": create_archive,
        **paths,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def run_customization(
    output_dir: Path,
    *,
    profile_path: Path | None = None,
    input_path: Path | None = None,
    template_id: str | None = None,
    applied_output: Path | None = None,
) -> int:
    profile = load_organization_profile(profile_path) if profile_path is not None else default_organization_profile()
    paths = write_customization_artifacts(output_dir, profile=profile, template_id=template_id)
    validation = validate_organization_profile(profile)
    if input_path is not None:
        cfg = load_input(input_path)
        customized = apply_organization_profile(cfg, profile, template_id=template_id)
        destination = applied_output or output_dir / "customized_input.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(customized, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths["customized_input"] = str(destination)
    print(
        json.dumps(
            {
                "passed": bool(validation.get("passed", False)),
                "error_count": int(validation.get("error_count", 0) or 0),
                "warning_count": int(validation.get("warning_count", 0) or 0),
                "out": str(output_dir),
                **paths,
            },
            ensure_ascii=False,
        )
    )
    return 0 if validation.get("passed", False) else 1


def load_input(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise FEM2DError("input root must be a mapping")
    cfg = dict(data)
    mesh = cfg.get("mesh")
    if isinstance(mesh, Mapping) and str(mesh.get("source", "")).strip().lower() in {"external", "file"} and "base_dir" not in mesh:
        updated_mesh = dict(mesh)
        updated_mesh["base_dir"] = str(path.resolve().parent)
        cfg["mesh"] = updated_mesh
    return cfg


def detect_dimension(cfg: Mapping[str, Any]) -> str:
    analysis = cfg.get("analysis", {})
    if isinstance(analysis, Mapping) and analysis.get("dimension"):
        return str(analysis["dimension"]).upper().replace(" ", "")
    model = cfg.get("model", {})
    if isinstance(model, Mapping) and model.get("dimension"):
        return str(model["dimension"]).upper().replace(" ", "")
    element = cfg.get("element", {})
    if isinstance(element, Mapping):
        etype = str(element.get("type", "")).upper()
        if etype in {"TRI3", "TRI6", "QUAD4", "QUAD8"}:
            return "2D"
    mesh = cfg.get("mesh", {})
    if isinstance(mesh, Mapping):
        etype = str(mesh.get("element_type", mesh.get("type", ""))).upper()
        if etype in {"TRI3", "TRI6", "QUAD4", "QUAD8"}:
            return "2D"
    return "2D"


def _default_run_dir(input_path: Path) -> str:
    return str(input_path.resolve().parent / "runs" / input_path.stem)


if __name__ == "__main__":
    raise SystemExit(main())
