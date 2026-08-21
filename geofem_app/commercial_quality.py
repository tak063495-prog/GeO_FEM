"""Commercial-grade quality gates for GeoFEM release and result validation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .geofeas_verification import audit_post_report_package
from .standard_benchmarks import run_standard_benchmark_suite
from .startup_check import run_startup_check


def run_output_reliability_gate(
    result_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate that one solve output is internally consistent and auditable."""

    root = Path(result_dir)
    checks: list[dict[str, Any]] = []
    summary = _read_json(root / "summary.json")
    index = _read_json(root / "result_view_index.json")
    manifest = _read_json(root / "case_manifest.json")
    diagnostics = _read_json(root / "input_diagnostics.json")
    report_manifest = _read_json(root / "calculation_report_manifest.json")

    _file_check(checks, root, "summary.json", "summary.json", "ERROR", "analysis summary")
    _file_check(checks, root, "result_view_index.json", "result_view_index.json", "ERROR", "result navigation index")
    _file_check(checks, root, "case_manifest.json", "case_manifest.json", "ERROR", "case manifest")
    _file_check(checks, root, "input_diagnostics.json", "input_diagnostics.json", "ERROR", "input diagnostics")
    _file_check(checks, root, "input_assistance.json", "input_assistance.json", "WARN", "input assistance templates and unit/range guidance")
    _file_check(checks, root, "input_assistance.csv", "input_assistance.csv", "WARN", "input assistance table")
    _file_check(checks, root, "input_assistance.html", "input_assistance.html", "WARN", "input assistance report")
    _file_check(checks, root, "calculation_report.html", "report.html", "ERROR", "HTML calculation report")
    _file_check(checks, root, "calculation_report.pdf", "report.pdf", "ERROR", "PDF calculation report")
    _file_check(checks, root, "calculation_report_manifest.json", "report.manifest", "ERROR", "report reproducibility manifest")
    _file_check(checks, root, "calculation_report_input_snapshot.json", "report.input_snapshot", "ERROR", "frozen input snapshot")
    _file_check(checks, root, "analysis_log.json", "analysis_log.json", "WARN", "structured analysis log")
    _file_check(checks, root, "performance_summary.json", "performance_summary.json", "WARN", "performance summary")
    _file_check(checks, root, "performance_kpi_matrix.json", "performance_kpi_matrix.json", "WARN", "commercial performance KPI matrix")
    _file_check(checks, root, "large_model_operations.json", "large_model_operations.json", "WARN", "large-model operation profile")
    _file_check(checks, root, "large_model_operations.csv", "large_model_operations.csv", "WARN", "large-model operation KPI table")
    _file_check(checks, root, "large_model_operations.html", "large_model_operations.html", "WARN", "large-model operation report")
    _file_check(checks, root, "large_model_node_index.csv", "large_model_node_index.csv", "WARN", "node search index")
    _file_check(checks, root, "large_model_element_index.csv", "large_model_element_index.csv", "WARN", "element search index")
    _file_check(checks, root, "mesh_quality.json", "mesh_quality.json", "WARN", "mesh quality report")
    _file_check(checks, root, "reliability_summary.json", "reliability_summary.json", "WARN", "human-readable reliability summary")
    _file_check(checks, root, "project_audit_trail.json", "project_audit_trail.json", "WARN", "project-level audit trail")

    _mapping_check(checks, "summary.readable", bool(summary), "ERROR", "summary JSON is readable")
    _mapping_check(checks, "result_index.readable", bool(index), "ERROR", "result index JSON is readable")
    _mapping_check(checks, "case_manifest.completed", manifest.get("status") == "completed", "ERROR", "case manifest status is completed", actual=manifest.get("status", "missing"))
    _mapping_check(
        checks,
        "input_diagnostics.no_errors",
        int(diagnostics.get("error_count", 0) or 0) == 0,
        "ERROR",
        "input diagnostics error count is zero",
        actual=diagnostics.get("error_count", "missing"),
    )
    assistance = _read_json(root / "input_assistance.json")
    assistance_features = assistance.get("features", []) if isinstance(assistance.get("features", []), list) else []
    for feature in ("analysis_templates", "material_templates", "boundary_templates", "field_units", "recommended_ranges", "prohibited_ranges", "immediate_config_diagnostics"):
        _mapping_check(
            checks,
            f"input_assistance.feature.{feature}",
            feature in assistance_features,
            "WARN",
            f"input assistance includes {feature}",
            actual=",".join(str(item) for item in assistance_features),
            expected=feature,
        )

    summary_stage_count = len(summary.get("stages", [])) if isinstance(summary.get("stages", []), list) else 0
    index_stage_count = int(index.get("stage_count", 0) or 0)
    _mapping_check(
        checks,
        "stage_count.match",
        bool(summary) and bool(index) and summary_stage_count == index_stage_count,
        "ERROR",
        "summary stage count matches result index stage count",
        actual=f"summary={summary_stage_count}, index={index_stage_count}",
        expected="equal non-conflicting stage counts",
    )

    for stage in _index_stages(index):
        _stage_table_checks(checks, stage)

    report_features = report_manifest.get("features", [])
    kpi_matrix = _read_json(root / "performance_kpi_matrix.json")
    coverage = kpi_matrix.get("area_coverage", {}) if isinstance(kpi_matrix.get("area_coverage", {}), Mapping) else {}
    for area in ("cold_run", "warm_solver", "solver_profile", "gui", "post", "report", "cad_mesh", "numba"):
        _mapping_check(
            checks,
            f"performance_kpi.area.{area}",
            bool(coverage.get(area, False)),
            "WARN",
            f"performance KPI matrix includes {area}",
            actual=coverage.get(area, "missing"),
            expected=True,
        )
    large_model = _read_json(root / "large_model_operations.json")
    large_model_features = large_model.get("features", [])
    for feature in (
        "node_search_index",
        "element_search_index",
        "display_lod_policy",
        "partial_selection_plan",
        "result_table_virtualization",
        "response_time_measurements",
    ):
        _mapping_check(
            checks,
            f"large_model_operations.feature.{feature}",
            feature in large_model_features if isinstance(large_model_features, list) else False,
            "WARN",
            f"large-model operation profile includes {feature}",
            actual=",".join(str(item) for item in large_model_features) if isinstance(large_model_features, list) else large_model_features,
            expected=feature,
        )
    _mapping_check(
        checks,
        "large_model_operations.response_time",
        bool(large_model.get("response_time", {}).get("passed")) if isinstance(large_model.get("response_time", {}), Mapping) else False,
        "WARN",
        "large-model operation response probes are within budgets",
        actual=large_model.get("response_time", {}).get("passed") if isinstance(large_model.get("response_time", {}), Mapping) else "missing",
        expected=True,
    )
    reliability = _read_json(root / "reliability_summary.json")
    _mapping_check(
        checks,
        "reliability_summary.passed",
        bool(reliability.get("passed", False)),
        "WARN",
        "reliability summary has no required failures",
        actual=reliability.get("passed", "missing"),
        expected=True,
    )
    reliability_features = reliability.get("features", [])
    for feature in ("convergence_status", "equilibrium_residual_and_boundary_reaction_summary", "energy_terms_when_available", "mass_balance_terms_when_available", "mesh_quality_summary", "input_hash"):
        _mapping_check(
            checks,
            f"reliability_summary.feature.{feature}",
            feature in reliability_features if isinstance(reliability_features, list) else False,
            "WARN",
            f"reliability summary includes {feature}",
            actual=",".join(str(item) for item in reliability_features) if isinstance(reliability_features, list) else reliability_features,
            expected=feature,
        )
    audit_trail = _read_json(root / "project_audit_trail.json")
    _mapping_check(
        checks,
        "project_audit_trail.passed",
        bool(audit_trail.get("passed", False)),
        "WARN",
        "project audit trail has no required failures",
        actual=audit_trail.get("passed", "missing"),
        expected=True,
    )
    audit_features = audit_trail.get("features", [])
    for feature in ("case_manifest_and_input_diff", "gui_operation_log_tail", "stage_approval_history", "output_artifact_hashes"):
        _mapping_check(
            checks,
            f"project_audit_trail.feature.{feature}",
            feature in audit_features if isinstance(audit_features, list) else False,
            "WARN",
            f"project audit trail includes {feature}",
            actual=",".join(str(item) for item in audit_features) if isinstance(audit_features, list) else audit_features,
            expected=feature,
        )
    _mapping_check(
        checks,
        "report.manifest.frozen",
        bool(report_manifest.get("reproducibility", {}).get("frozen")) if isinstance(report_manifest.get("reproducibility", {}), Mapping) else False,
        "ERROR",
        "report manifest marks inputs as frozen",
        actual=report_manifest.get("reproducibility", {}).get("frozen") if isinstance(report_manifest.get("reproducibility", {}), Mapping) else "missing",
        expected=True,
    )
    input_hash = ""
    if isinstance(report_manifest.get("reproducibility", {}), Mapping):
        input_hash = str(report_manifest["reproducibility"].get("input_sha256", ""))
    _mapping_check(
        checks,
        "report.manifest.input_hash",
        len(input_hash) == 64,
        "ERROR",
        "report manifest stores a 64-character input hash",
        actual=len(input_hash),
        expected=64,
    )
    _mapping_check(
        checks,
        "report.manifest.direct_pdf",
        "direct_pdf" in report_features if isinstance(report_features, list) else False,
        "WARN",
        "direct PDF report feature is recorded",
        actual=",".join(str(item) for item in report_features) if isinstance(report_features, list) else report_features,
        expected="direct_pdf",
    )

    errors = [row for row in checks if row["severity"] == "ERROR" and not bool(row["passed"])]
    warnings = [row for row in checks if row["severity"] == "WARN" and not bool(row["passed"])]
    payload = {
        "schema": "geofem.output_reliability_gate.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(root),
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "stage_count": summary_stage_count,
        "features": [
            "input_hash_and_manifest_gate",
            "stage_table_consistency_gate",
            "report_reproducibility_gate",
            "post_and_report_artifact_presence_gate",
            "commercial_performance_kpi_matrix_gate",
            "large_model_operation_gate",
            "human_readable_reliability_summary_gate",
            "project_audit_trail_gate",
        ],
        "checks": checks,
    }
    if output_dir is not None:
        _write_quality_artifacts(payload, Path(output_dir), stem="output_reliability_gate", title="GeoFEM Output Reliability Gate")
    return payload


def run_commercial_quality_check(
    output_dir: str | Path,
    *,
    result_dir: str | Path | None = None,
    baseline_dir: str | Path | None = None,
    run_benchmarks: bool = True,
    benchmark_baseline: str | Path | Mapping[str, Any] | None = None,
    max_slowdown: float = 1.25,
    include_startup: bool = False,
    include_gui: bool = False,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    """Run the product-quality gate that bundles result, report, and performance checks."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    modules: list[dict[str, Any]] = []
    if result_dir is not None:
        result_gate = run_output_reliability_gate(result_dir, output_dir=out / "output_reliability")
        modules.append(_module("output_reliability", "出力信頼性ゲート", result_gate, required=True, artifact_dir=out / "output_reliability"))
        post_audit = audit_post_report_package(result_dir, baseline_dir=baseline_dir, output_dir=out / "post_report_audit")
        modules.append(_module("post_report_audit", "Post/帳票監査", post_audit, required=False, artifact_dir=out / "post_report_audit"))
    else:
        modules.append(_skipped_module("output_reliability", "出力信頼性ゲート", "--result was not supplied"))
        modules.append(_skipped_module("post_report_audit", "Post/帳票監査", "--result was not supplied"))

    if run_benchmarks:
        benchmark = run_standard_benchmark_suite(out / "standard_benchmarks", baseline=benchmark_baseline, max_slowdown=max_slowdown)
        modules.append(_module("standard_benchmarks", "標準ベンチ/性能回帰", benchmark, required=True, artifact_dir=out / "standard_benchmarks"))
    else:
        modules.append(_skipped_module("standard_benchmarks", "標準ベンチ/性能回帰", "--skip-benchmarks"))

    if include_startup:
        startup = run_startup_check(out / "startup_check", include_gui=include_gui, run_sample=True)
        modules.append(_module("startup_check", "起動診断", startup, required=True, artifact_dir=out / "startup_check"))
    else:
        modules.append(_skipped_module("startup_check", "起動診断", "--include-startup was not supplied"))

    failed_required = [row for row in modules if row["required"] and row["status"] == "failed"]
    warning_modules = [row for row in modules if row["status"] == "warning"]
    payload = {
        "schema": "geofem.commercial_quality_check.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failed_required and (not warning_modules or not fail_on_warning),
        "failed_required_count": len(failed_required),
        "warning_module_count": len(warning_modules),
        "fail_on_warning": bool(fail_on_warning),
        "features": [
            "one_command_product_quality_gate",
            "output_reliability_gate",
            "post_report_audit_gate",
            "standard_benchmark_performance_gate",
            "startup_environment_gate",
        ],
        "modules": modules,
    }
    _write_quality_artifacts(payload, out, stem="commercial_quality_check", title="GeoFEM Commercial Quality Check")
    return payload


def _stage_table_checks(checks: list[dict[str, Any]], stage: Mapping[str, Any]) -> None:
    stage_name = str(stage.get("stage", stage.get("name", "")) or "stage")
    output_dir = Path(str(stage.get("output_dir", "")))
    _mapping_check(
        checks,
        f"stage.{stage_name}.directory",
        output_dir.exists(),
        "ERROR",
        f"stage output directory exists for {stage_name}",
        actual=str(output_dir),
        expected="existing directory",
    )
    for group in ("node_tables", "element_tables", "history_tables"):
        for table_name in _table_names(stage.get(group, "")):
            path = output_dir / f"{table_name}.csv"
            _mapping_check(
                checks,
                f"stage.{stage_name}.{table_name}",
                path.exists(),
                "ERROR",
                f"result table exists: {stage_name}/{table_name}.csv",
                actual=str(path),
                expected="existing CSV",
            )


def _index_stages(index: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    stages = index.get("stages", [])
    if not isinstance(stages, list):
        return []
    return [stage for stage in stages if isinstance(stage, Mapping)]


def _table_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _file_check(checks: list[dict[str, Any]], root: Path, relative: str, check_id: str, severity: str, detail: str) -> None:
    path = root / relative
    checks.append(
        {
            "id": check_id,
            "severity": severity,
            "passed": path.exists(),
            "actual": str(path),
            "expected": "existing file",
            "detail": detail,
        }
    )


def _mapping_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    severity: str,
    detail: str,
    *,
    actual: Any = True,
    expected: Any = True,
) -> None:
    checks.append(
        {
            "id": check_id,
            "severity": severity,
            "passed": bool(passed),
            "actual": actual,
            "expected": expected,
            "detail": detail,
        }
    )


def _module(name: str, label: str, summary: Mapping[str, Any], *, required: bool, artifact_dir: Path) -> dict[str, Any]:
    passed = bool(summary.get("passed", False))
    error_count = int(summary.get("error_count", summary.get("failed_count", 0)) or 0)
    warning_count = int(summary.get("warning_count", summary.get("performance_regression_count", 0)) or 0)
    status = "passed" if passed else ("failed" if required else "warning")
    return {
        "name": name,
        "label": label,
        "required": required,
        "status": status,
        "passed": passed,
        "error_count": error_count,
        "warning_count": warning_count,
        "artifact_dir": str(artifact_dir),
        "schema": summary.get("schema", ""),
    }


def _skipped_module(name: str, label: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "required": False,
        "status": "skipped",
        "passed": True,
        "error_count": 0,
        "warning_count": 0,
        "artifact_dir": "",
        "schema": "",
        "skipped_reason": reason,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _write_quality_artifacts(payload: Mapping[str, Any], out: Path, *, stem: str, title: str) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    csv_path = out / f"{stem}.csv"
    html_path = out / f"{stem}.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_rows_csv(payload, csv_path)
    html_path.write_text(_quality_html(payload, title), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _write_rows_csv(payload: Mapping[str, Any], path: Path) -> None:
    rows = payload.get("checks", payload.get("modules", []))
    fields = ["id", "name", "label", "severity", "status", "required", "passed", "actual", "expected", "detail", "artifact_dir", "schema"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    writer.writerow({field: row.get(field, "") for field in fields})


def _quality_html(payload: Mapping[str, Any], title: str) -> str:
    rows = payload.get("checks", payload.get("modules", []))
    table_rows = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                key = row.get("id", row.get("name", ""))
                status = row.get("status", row.get("severity", ""))
                passed = row.get("passed", "")
                detail = row.get("detail", row.get("label", ""))
                table_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(key))}</td>"
                    f"<td>{html.escape(str(status))}</td>"
                    f"<td>{html.escape(str(passed))}</td>"
                    f"<td>{html.escape(str(detail))}</td>"
                    "</tr>"
                )
    body = "\n".join(table_rows) or "<tr><td colspan='4'>No checks.</td></tr>"
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Meiryo,Arial,sans-serif;line-height:1.5}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:5px}th{background:#eef2f7}.ok{color:#166534}.ng{color:#991b1b}</style>"
        f"<body><h1>{html.escape(title)}</h1>"
        f"<p>passed=<strong>{html.escape(str(payload.get('passed', '')))}</strong></p>"
        "<table><tr><th>id/name</th><th>severity/status</th><th>passed</th><th>detail</th></tr>"
        f"{body}</table></body></html>"
    )


__all__ = ["run_commercial_quality_check", "run_output_reliability_gate"]
