"""Run case manifests and recovery reports for GeoFEM analyses."""

from __future__ import annotations

from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import platform
import traceback
from typing import Any, Mapping

from .audit_trail import write_project_audit_trail
from .failure_diagnostics import classify_failure, write_failure_diagnostics
from .failure_recovery import build_failure_recovery_plan, write_failure_recovery_plan
from .project import GeoFEMProject, add_run_record, load_project, save_project, update_after_run


def input_file_hash(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def write_case_manifest(
    *,
    output_dir: str | Path,
    input_path: str | Path,
    diagnostics: Mapping[str, Any] | None,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    elapsed_seconds: float | None = None,
    result_summary: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    current_hash = input_file_hash(input_path) if input_path.exists() else ""
    previous = _latest_history_record(out.parent)
    diff = {
        "schema": "geofem.input_diff.v1",
        "current_input": str(input_path),
        "current_hash": current_hash,
        "previous_hash": previous.get("input_hash", "") if previous else "",
        "changed_since_previous": bool(previous and previous.get("input_hash") != current_hash),
        "previous_output_dir": previous.get("output_dir", "") if previous else "",
    }
    (out / "input_diff.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema": "geofem.case_manifest.v1",
        "status": status,
        "input_file": str(input_path),
        "input_hash": current_hash,
        "output_dir": str(out),
        "started_at": started_at,
        "finished_at": finished_at or datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "diagnostics": {
            "passed": bool((diagnostics or {}).get("passed", False)),
            "error_count": int((diagnostics or {}).get("error_count", 0) or 0),
            "warning_count": int((diagnostics or {}).get("warning_count", 0) or 0),
            "json": str(out / "input_diagnostics.json"),
            "html": str(out / "input_diagnostics.html"),
        },
        "input_diff": str(out / "input_diff.json"),
        "summary": str(out / "summary.json") if (out / "summary.json").exists() else "",
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
        },
    }
    if result_summary:
        manifest["result"] = dict(result_summary)
    if error is not None:
        failure_analysis = classify_failure(error, diagnostics=diagnostics)
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "analysis": {
                "category": failure_analysis.get("primary_category", ""),
                "cause": failure_analysis.get("primary_cause", ""),
                "stage": failure_analysis.get("primary_stage", ""),
                "target_id": failure_analysis.get("primary_target_id", ""),
                "input_path": failure_analysis.get("primary_input_path", ""),
                "gui_panel": failure_analysis.get("primary_gui_panel", ""),
                "recommended_fix": failure_analysis.get("primary_recommended_fix", ""),
                "rerun_condition": failure_analysis.get("primary_rerun_condition", ""),
                "json": str(out / "failure_diagnostics.json") if (out / "failure_diagnostics.json").exists() else "",
                "recovery_json": str(out / "failure_recovery_plan.json") if (out / "failure_recovery_plan.json").exists() else "",
            },
        }
    manifest_path = out / "case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try:
        manifest["audit_trail"] = write_project_audit_trail(out, project_root=input_path.parent, input_path=input_path)
    except Exception as exc:
        manifest["audit_trail_error"] = str(exc)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _append_history(out.parent / "case_history.jsonl", manifest)
    _update_nearby_project(input_path, manifest)
    return manifest


def write_failure_report(
    *,
    output_dir: str | Path,
    input_path: str | Path,
    error: BaseException,
    diagnostics: Mapping[str, Any] | None = None,
    started_at: str | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failure_analysis = classify_failure(error, diagnostics=diagnostics)
    diagnostic_paths = write_failure_diagnostics(failure_analysis, out)
    recovery_plan = build_failure_recovery_plan(failure_analysis, output_dir=out, diagnostics=diagnostics)
    recovery_paths = write_failure_recovery_plan(recovery_plan, out)
    data = {
        "schema": "geofem.failure_report.v1",
        "input_file": str(input_path),
        "output_dir": str(out),
        "started_at": started_at or "",
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "error": {"type": type(error).__name__, "message": str(error)},
        "failure_analysis": failure_analysis,
        "failure_diagnostics": diagnostic_paths,
        "failure_recovery": recovery_plan,
        "failure_recovery_paths": recovery_paths,
        "diagnostics": dict(diagnostics or {}),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        "artifacts": sorted(path.name for path in out.iterdir()) if out.exists() else [],
    }
    json_path = out / "failure_report.json"
    html_path = out / "failure_report.html"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html_path.write_text(_failure_html(data), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}


def _append_history(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, default=str) + "\n")


def _latest_history_record(root: Path) -> dict[str, Any]:
    path = root / "case_history.jsonl"
    if not path.exists():
        return {}
    last: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            last = row
    return last


def _update_nearby_project(input_path: Path, manifest: Mapping[str, Any]) -> None:
    for folder in [input_path.parent, *input_path.parents]:
        candidates = sorted(folder.glob("*.gfemproj"))
        if not candidates:
            continue
        try:
            project: GeoFEMProject = load_project(candidates[0])
            project.input_file = str(input_path)
            update_after_run(project, str(manifest.get("output_dir", "")))
            add_run_record(project, manifest)
            save_project(project, candidates[0])
        except Exception:
            return
        return


def _failure_html(data: Mapping[str, Any]) -> str:
    error = data.get("error", {}) if isinstance(data.get("error", {}), Mapping) else {}
    analysis = data.get("failure_analysis", {}) if isinstance(data.get("failure_analysis", {}), Mapping) else {}
    recovery = data.get("failure_recovery", {}) if isinstance(data.get("failure_recovery", {}), Mapping) else {}
    trace = "\n".join(str(line) for line in data.get("traceback", []))
    rows = []
    for row in analysis.get("findings", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('severity', '')))}</td>"
            f"<td>{html.escape(str(row.get('category', '')))}</td>"
            f"<td>{html.escape(str(row.get('cause', '')))}</td>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{html.escape(str(row.get('target_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('input_path', '')))}</td>"
            f"<td>{html.escape(str(row.get('gui_panel', '')))}</td>"
            f"<td>{html.escape(str(row.get('recommended_fix', '')))}</td>"
            f"<td>{html.escape(str(row.get('rerun_condition', '')))}</td>"
            "</tr>"
        )
    recovery_rows = []
    for action in recovery.get("actions", []):
        if not isinstance(action, Mapping):
            continue
        recovery_rows.append(
            "<tr>"
            f"<td>{html.escape(str(action.get('rank', '')))}</td>"
            f"<td>{html.escape(str(action.get('priority', '')))}</td>"
            f"<td>{html.escape(str(action.get('title', '')))}</td>"
            f"<td>{html.escape(str(action.get('target_panel', '')))}</td>"
            f"<td>{html.escape(str(action.get('target_path', '')))}</td>"
            f"<td>{html.escape(str(action.get('action', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM failure report</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}pre{{background:#f6f6f6;padding:12px;overflow:auto}}dt{{font-weight:bold}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#f2f2f2}}</style></head>
<body><h1>解析失敗レポート</h1>
<dl>
<dt>入力</dt><dd>{html.escape(str(data.get('input_file', '')))}</dd>
<dt>出力</dt><dd>{html.escape(str(data.get('output_dir', '')))}</dd>
<dt>エラー</dt><dd>{html.escape(str(error.get('type', '')))}: {html.escape(str(error.get('message', '')))}</dd>
<dt>主原因</dt><dd>{html.escape(str(analysis.get('primary_cause', '')))}</dd>
<dt>該当GUI画面</dt><dd>{html.escape(str(analysis.get('primary_gui_panel', '')))}</dd>
<dt>推奨修正</dt><dd>{html.escape(str(analysis.get('primary_recommended_fix', '')))}</dd>
<dt>再実行条件</dt><dd>{html.escape(str(analysis.get('primary_rerun_condition', '')))}</dd>
</dl>
<h2>エラー分類</h2>
<table><thead><tr><th>severity</th><th>category</th><th>cause</th><th>stage</th><th>target</th><th>input path</th><th>GUI panel</th><th>recommended fix</th><th>rerun condition</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>復旧候補</h2>
<table><thead><tr><th>rank</th><th>priority</th><th>title</th><th>GUI panel</th><th>target</th><th>action</th></tr></thead><tbody>{''.join(recovery_rows)}</tbody></table>
<h2>トレースバック</h2><pre>{html.escape(trace)}</pre>
</body></html>
"""


__all__ = ["input_file_hash", "write_case_manifest", "write_failure_report"]
