"""Project-level audit trail artifacts for solved or failed cases."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping


AUDIT_ARTIFACT_STEMS = {"project_audit_trail"}


def build_project_audit_trail(
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    input_path: str | Path | None = None,
    max_operation_events: int = 200,
) -> dict[str, Any]:
    root = Path(output_dir)
    project = Path(project_root) if project_root is not None else root.parent
    input_file = Path(input_path) if input_path is not None else None
    case_manifest = _read_json(root / "case_manifest.json")
    summary = _read_json(root / "summary.json")
    input_diff = _read_json(root / "input_diff.json")
    diagnostics = _read_json(root / "input_diagnostics.json")
    report_manifest = _read_json(root / "calculation_report_manifest.json")
    reliability = _read_json(root / "reliability_summary.json")
    performance = _read_json(root / "performance_summary.json")
    kpis = _read_json(root / "performance_kpi_matrix.json")
    failure = _read_json(root / "failure_report.json")
    input_snapshot = _read_json(root / "calculation_report_input_snapshot.json")
    operations = _read_jsonl_tail(project / ".geofem_audit_log.jsonl", max_operation_events)
    history = _read_jsonl_tail(root.parent / "case_history.jsonl", 50)
    approvals = _approval_rows(input_snapshot)
    artifacts = _artifact_hash_rows(root)
    events = _audit_events(
        case_manifest=case_manifest,
        input_diff=input_diff,
        diagnostics=diagnostics,
        report_manifest=report_manifest,
        reliability=reliability,
        performance=performance,
        kpis=kpis,
        failure=failure,
        operations=operations,
        approvals=approvals,
        artifacts=artifacts,
    )
    checks = _audit_checks(case_manifest, input_diff, diagnostics, report_manifest, reliability, artifacts)
    failed_required = [row for row in checks if row["severity"] == "ERROR" and not row["passed"]]
    warnings = [row for row in checks if row["severity"] == "WARN" and not row["passed"]]
    return {
        "schema": "geofem.project_audit_trail.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project),
        "output_dir": str(root),
        "input_file": str(input_file or case_manifest.get("input_file", "")),
        "status": case_manifest.get("status", "unknown"),
        "passed": not failed_required,
        "error_count": len(failed_required),
        "warning_count": len(warnings),
        "features": [
            "case_manifest_and_input_diff",
            "input_diagnostics_summary",
            "execution_result_summary",
            "report_reproducibility_manifest",
            "reliability_and_performance_summaries",
            "gui_operation_log_tail",
            "stage_approval_history",
            "output_artifact_hashes",
            "project_case_history_tail",
        ],
        "counts": {
            "operation_event_count": len(operations),
            "approval_count": len(approvals),
            "artifact_count": len(artifacts),
            "case_history_count": len(history),
            "event_count": len(events),
            "check_count": len(checks),
        },
        "case_manifest": _compact_manifest(case_manifest),
        "input_diff": input_diff,
        "diagnostics": _diagnostics_summary(diagnostics),
        "summary": _summary_overview(summary),
        "report_reproducibility": _report_reproducibility(report_manifest),
        "reliability": _reliability_overview(reliability),
        "performance": _performance_overview(performance, kpis),
        "failure": _failure_overview(failure),
        "operations": operations,
        "approvals": approvals,
        "artifacts": artifacts,
        "case_history_tail": history,
        "checks": checks,
        "events": events,
    }


def write_project_audit_trail(
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    input_path: str | Path | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = build_project_audit_trail(out, project_root=project_root, input_path=input_path)
    json_path = out / "project_audit_trail.json"
    csv_path = out / "project_audit_trail.csv"
    html_path = out / "project_audit_trail.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_audit_csv(payload, csv_path)
    html_path.write_text(_audit_html(payload), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _audit_events(**sections: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    manifest = _mapping(sections.get("case_manifest"))
    if manifest:
        events.append(
            {
                "kind": "analysis_execution",
                "time": manifest.get("finished_at", manifest.get("started_at", "")),
                "target": manifest.get("output_dir", ""),
                "detail": f"status={manifest.get('status', '')}, elapsed={manifest.get('elapsed_seconds', '')}",
            }
        )
    diff = _mapping(sections.get("input_diff"))
    if diff:
        events.append({"kind": "input_diff", "time": "", "target": diff.get("current_input", ""), "detail": f"changed={diff.get('changed_since_previous', '')}"})
    diagnostics = _mapping(sections.get("diagnostics"))
    if diagnostics:
        events.append(
            {
                "kind": "input_diagnostics",
                "time": "",
                "target": diagnostics.get("json", "input_diagnostics.json"),
                "detail": f"errors={diagnostics.get('error_count', '')}, warnings={diagnostics.get('warning_count', '')}",
            }
        )
    report = _mapping(sections.get("report_manifest"))
    if report:
        repro = _mapping(report.get("reproducibility"))
        events.append({"kind": "report_freeze", "time": report.get("generated_at", ""), "target": "calculation_report_manifest.json", "detail": f"frozen={repro.get('frozen', '')}"})
    reliability = _mapping(sections.get("reliability"))
    if reliability:
        events.append({"kind": "reliability_summary", "time": "", "target": "reliability_summary.json", "detail": f"passed={reliability.get('passed', '')}"})
    performance = _mapping(sections.get("performance"))
    if performance:
        events.append({"kind": "performance_summary", "time": "", "target": "performance_summary.json", "detail": f"elapsed={performance.get('elapsed_seconds', '')}"})
    kpis = _mapping(sections.get("kpis"))
    if kpis:
        events.append({"kind": "performance_kpi_matrix", "time": "", "target": "performance_kpi_matrix.json", "detail": f"measured={kpis.get('measured_count', '')}"})
    failure = _mapping(sections.get("failure"))
    if failure:
        error = _mapping(failure.get("error"))
        events.append({"kind": "failure_report", "time": failure.get("failed_at", ""), "target": "failure_report.json", "detail": f"{error.get('type', '')}: {error.get('message', '')}"})
    for row in sections.get("operations", []):
        if isinstance(row, Mapping):
            events.append({"kind": "gui_operation", "time": row.get("time", row.get("timestamp", "")), "target": row.get("target", ""), "detail": row.get("action", row.get("event", ""))})
    for row in sections.get("approvals", []):
        if isinstance(row, Mapping):
            events.append({"kind": "stage_approval", "time": row.get("at", row.get("timestamp", "")), "target": row.get("approval_key", row.get("diff", "")), "detail": row.get("status", row.get("action", ""))})
    for row in sections.get("artifacts", []):
        if isinstance(row, Mapping):
            events.append({"kind": "artifact_hash", "time": "", "target": row.get("path", ""), "detail": f"sha256={str(row.get('sha256', ''))[:12]}"})
    return events


def _audit_checks(
    manifest: Mapping[str, Any],
    input_diff: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    report_manifest: Mapping[str, Any],
    reliability: Mapping[str, Any],
    artifacts: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _add_check(checks, "case_manifest.present", bool(manifest), "ERROR", "case_manifest.json を監査証跡に統合している")
    _add_check(checks, "input_diff.present", bool(input_diff), "WARN", "input_diff.json を監査証跡に統合している")
    _add_check(checks, "diagnostics.present", bool(diagnostics), "WARN", "input_diagnostics.json を監査証跡に統合している")
    _add_check(checks, "report_manifest.present", bool(report_manifest), "WARN", "calculation_report_manifest.json を監査証跡に統合している")
    _add_check(checks, "reliability.present", bool(reliability), "WARN", "reliability_summary.json を監査証跡に統合している")
    _add_check(checks, "artifacts.has_hashes", len(artifacts) > 0 and all(row.get("sha256") for row in artifacts), "ERROR", "出力物ハッシュを記録している")
    return checks


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, severity: str, detail: str) -> None:
    checks.append({"id": check_id, "severity": severity, "passed": bool(passed), "detail": detail})


def _approval_rows(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = snapshot
    if isinstance(snapshot.get("input_config"), Mapping):
        source = snapshot["input_config"]  # type: ignore[assignment]
    rows: list[dict[str, Any]] = []
    approvals = source.get("stage_diff_approvals", {}) if isinstance(source, Mapping) else {}
    if isinstance(approvals, Mapping):
        for key, value in approvals.items():
            record = dict(value) if isinstance(value, Mapping) else {"status": value}
            rows.append({"approval_key": str(key), **record})
    history = source.get("stage_diff_approval_history", []) if isinstance(source, Mapping) else []
    if isinstance(history, list):
        for value in history:
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _artifact_hash_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.stem in AUDIT_ARTIFACT_STEMS:
            continue
        rel = path.relative_to(root).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": _file_sha256(path), "role": _artifact_role(path.name)})
    return rows


def _artifact_role(name: str) -> str:
    lowered = name.lower()
    if "manifest" in lowered:
        return "manifest"
    if "report" in lowered or lowered.endswith(".pdf"):
        return "report"
    if lowered.endswith(".csv"):
        return "table"
    if lowered.endswith(".json"):
        return "structured-data"
    if lowered.endswith((".html", ".htm")):
        return "html-view"
    return "artifact"


def _compact_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key, "") for key in ("schema", "status", "input_file", "input_hash", "output_dir", "started_at", "finished_at", "elapsed_seconds", "summary")}


def _diagnostics_summary(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: diagnostics.get(key, "") for key in ("passed", "error_count", "warning_count", "issue_count")}


def _summary_overview(summary: Mapping[str, Any]) -> dict[str, Any]:
    stages = summary.get("stages", [])
    return {
        "schema": summary.get("schema", ""),
        "stage_count": len(stages) if isinstance(stages, list) else "",
        "report": summary.get("report", ""),
        "report_pdf": summary.get("report_pdf", ""),
    }


def _report_reproducibility(report_manifest: Mapping[str, Any]) -> dict[str, Any]:
    repro = _mapping(report_manifest.get("reproducibility"))
    return {
        "frozen": repro.get("frozen", ""),
        "input_sha256": repro.get("input_sha256", ""),
        "summary_sha256": repro.get("summary_sha256", ""),
        "html_sha256": repro.get("html_sha256", ""),
        "pdf_sha256": repro.get("pdf_sha256", ""),
    }


def _reliability_overview(reliability: Mapping[str, Any]) -> dict[str, Any]:
    return {key: reliability.get(key, "") for key in ("passed", "error_count", "warning_count")}


def _performance_overview(performance: Mapping[str, Any], kpis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "elapsed_seconds": performance.get("elapsed_seconds", ""),
        "stage_count": performance.get("stage_count", ""),
        "total_solver_iterations": performance.get("total_solver_iterations", ""),
        "kpi_measured_count": kpis.get("measured_count", ""),
        "kpi_warning_count": kpis.get("warning_count", ""),
    }


def _failure_overview(failure: Mapping[str, Any]) -> dict[str, Any]:
    error = _mapping(failure.get("error"))
    analysis = _mapping(failure.get("failure_analysis"))
    return {
        "type": error.get("type", ""),
        "message": error.get("message", ""),
        "category": analysis.get("primary_category", ""),
        "gui_panel": analysis.get("primary_gui_panel", ""),
        "recommended_fix": analysis.get("primary_recommended_fix", ""),
    }


def _write_audit_csv(payload: Mapping[str, Any], path: Path) -> None:
    fields = ["section", "kind", "id", "time", "target", "detail", "severity", "passed", "path", "bytes", "sha256", "role"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("events", []):
            if isinstance(row, Mapping):
                writer.writerow({"section": "event", **{field: row.get(field, "") for field in fields}})
        for row in payload.get("checks", []):
            if isinstance(row, Mapping):
                writer.writerow({"section": "check", **{field: row.get(field, "") for field in fields}})
        for row in payload.get("artifacts", []):
            if isinstance(row, Mapping):
                writer.writerow({"section": "artifact", **{field: row.get(field, "") for field in fields}})


def _audit_html(payload: Mapping[str, Any]) -> str:
    events = _html_rows(payload.get("events", []), ("kind", "time", "target", "detail"))
    checks = _html_rows(payload.get("checks", []), ("id", "severity", "passed", "detail"))
    artifacts = _html_rows(payload.get("artifacts", [])[:200] if isinstance(payload.get("artifacts"), list) else [], ("path", "role", "bytes", "sha256"))
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM Project Audit Trail</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#f3f3f3}}code{{word-break:break-all}}</style></head>
<body><h1>プロジェクト監査証跡</h1>
<p>status={html.escape(str(payload.get('status', '')))}, passed={html.escape(str(payload.get('passed', '')))}, output={html.escape(str(payload.get('output_dir', '')))}</p>
<h2>確認項目</h2>{checks}
<h2>監査イベント</h2>{events}
<h2>出力物ハッシュ</h2>{artifacts}
</body></html>
"""


def _html_rows(rows_data: Any, columns: tuple[str, ...]) -> str:
    rows = []
    if isinstance(rows_data, list):
        for row in rows_data:
            if not isinstance(row, Mapping):
                continue
            cells = "".join(f"<td><code>{html.escape(str(row.get(column, '')))}</code></td>" for column in columns)
            rows.append(f"<tr>{cells}</tr>")
    heads = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, Mapping):
                rows.append(dict(data))
    except Exception:
        return []
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["build_project_audit_trail", "write_project_audit_trail"]
