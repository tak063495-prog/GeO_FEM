"""Workspace dashboard and archive helpers for GeoFEM projects."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile
from typing import Any, Iterable, Mapping, Sequence

from .html_report_utils import html_escape, report_css, table
from .project import PROJECT_EXTENSION, PROJECT_DIRS
from .public_artifacts import write_dict_rows_csv, write_json_artifact


WORKSPACE_DASHBOARD_SCHEMA = "geofem.workspace_dashboard.v1"
WORKSPACE_ARCHIVE_SCHEMA = "geofem.workspace_archive.v1"
ARTIFACT_ROOTS = ("runs", "results", "reports", "logs", "input", "templates", "autosave", "post_baselines")
EXCLUDED_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
ARCHIVE_EXCLUDED_SUFFIXES = {".zip"}


def build_workspace_dashboard(
    project_root: str | Path,
    *,
    recent_projects: Sequence[str] | None = None,
    storage_warning_bytes: int = 1_000_000_000,
    artifact_row_limit: int = 500,
) -> dict[str, Any]:
    """Build a compact commercial-style workspace overview."""

    root = Path(project_root)
    project_files = _project_file_rows(root)
    run_rows = _run_rows(root)
    all_artifact_rows = _artifact_rows(root)
    artifact_rows = all_artifact_rows[:artifact_row_limit]
    storage = _storage_summary(root, artifact_rows=all_artifact_rows, warning_bytes=storage_warning_bytes)
    recent_inputs = _recent_input_rows(root, recent_projects=recent_projects)
    archive_plan = _archive_plan(root, storage)
    return {
        "schema": WORKSPACE_DASHBOARD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "passed": True,
        "features": [
            "project_dashboard",
            "recent_analysis_history",
            "artifact_inventory",
            "storage_management",
            "workspace_archive",
        ],
        "counts": {
            "project_file_count": len(project_files),
            "recent_input_count": len(recent_inputs),
            "run_count": len(run_rows),
            "artifact_count": len(all_artifact_rows),
            "artifact_list_count": len(artifact_rows),
            "archive_candidate_count": len(archive_plan["candidate_roots"]),
        },
        "projects": project_files,
        "recent_inputs": recent_inputs,
        "runs": run_rows,
        "artifacts": artifact_rows,
        "storage": storage,
        "archive_plan": archive_plan,
    }


def write_workspace_dashboard(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    recent_projects: Sequence[str] | None = None,
    storage_warning_bytes: int = 1_000_000_000,
    create_archive: bool = False,
    archive_name: str = "workspace_archive.zip",
    artifact_row_limit: int = 500,
) -> dict[str, str]:
    """Write JSON/CSV/HTML workspace dashboard artifacts and optional ZIP archive."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dashboard = build_workspace_dashboard(
        project_root,
        recent_projects=recent_projects,
        storage_warning_bytes=storage_warning_bytes,
        artifact_row_limit=artifact_row_limit,
    )
    json_path = out / "workspace_dashboard.json"
    runs_csv = out / "workspace_recent_runs.csv"
    artifacts_csv = out / "workspace_artifacts.csv"
    storage_csv = out / "workspace_storage.csv"
    html_path = out / "workspace_dashboard.html"
    write_json_artifact(json_path, dashboard)
    write_dict_rows_csv(
        runs_csv,
        dashboard["runs"],
        ["run_id", "status", "started_at", "finished_at", "elapsed_seconds", "input_file", "output_dir", "stage_count", "warning_count"],
    )
    write_dict_rows_csv(
        artifacts_csv,
        dashboard["artifacts"],
        ["path", "role", "area", "bytes", "modified_at", "run_id"],
    )
    write_dict_rows_csv(
        storage_csv,
        dashboard["storage"]["areas"],
        ["area", "bytes", "file_count", "warning"],
    )
    html_path.write_text(_dashboard_html(dashboard), encoding="utf-8")
    paths = {
        "json": str(json_path),
        "runs_csv": str(runs_csv),
        "artifacts_csv": str(artifacts_csv),
        "storage_csv": str(storage_csv),
        "html": str(html_path),
    }
    if create_archive:
        archive_path = out / archive_name
        manifest = archive_workspace(project_root, archive_path, dashboard=dashboard, skip_roots=[out])
        archive_manifest_path = out / "workspace_archive_manifest.json"
        write_json_artifact(archive_manifest_path, manifest)
        paths["archive"] = str(archive_path)
        paths["archive_manifest"] = str(archive_manifest_path)
    return paths


def archive_workspace(
    project_root: str | Path,
    archive_path: str | Path,
    *,
    dashboard: Mapping[str, Any] | None = None,
    skip_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Create a portable workspace archive ZIP with a manifest."""

    root = Path(project_root).resolve()
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    skip_resolved = [Path(path).resolve() for path in skip_roots]
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in _iter_archive_files(root):
            resolved = path.resolve()
            if resolved == archive.resolve() or any(_is_relative_to(resolved, skip) for skip in skip_resolved):
                continue
            rel = resolved.relative_to(root)
            zf.write(resolved, rel.as_posix())
            stat = resolved.stat()
            rows.append({"path": rel.as_posix(), "bytes": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        manifest = {
            "schema": WORKSPACE_ARCHIVE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(root),
            "archive_path": str(archive),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "dashboard_schema": str((dashboard or {}).get("schema", "")),
            "files": rows,
        }
        zf.writestr("workspace_archive_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return manifest


def _project_file_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"*{PROJECT_EXTENSION}")):
        data = _read_json(path)
        stat = path.stat()
        rows.append(
            {
                "path": _rel(root, path),
                "name": str(data.get("name", path.stem)),
                "dimension": str(data.get("dimension", "2D")),
                "unit_system": str(data.get("unit_system", "")),
                "analysis_type": str(data.get("analysis_type", "")),
                "latest_run": str(data.get("latest_run", "")),
                "run_record_count": len(data.get("run_records", [])) if isinstance(data.get("run_records"), list) else 0,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return rows


def _recent_input_rows(root: Path, *, recent_projects: Sequence[str] | None) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    if recent_projects:
        candidates.extend(Path(item) for item in recent_projects)
    input_dir = root / "input"
    for suffix in ("*.yaml", "*.yml", "*.json"):
        candidates.extend(sorted(input_dir.glob(suffix)))
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in candidates:
        normalized = str(path)
        if normalized in seen or not path.exists() or not path.is_file():
            continue
        seen.add(normalized)
        stat = path.stat()
        rows.append({"path": _rel(root, path), "bytes": stat.st_size, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    rows.sort(key=lambda row: str(row["modified_at"]), reverse=True)
    return rows[:24]


def _run_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    for manifest in _case_manifests(root):
        output = str(manifest.get("output_dir", ""))
        if output in seen_outputs:
            continue
        seen_outputs.add(output)
        output_path = Path(output) if output else Path()
        run_id = output_path.name if output else ""
        result = manifest.get("result", {}) if isinstance(manifest.get("result"), Mapping) else {}
        rows.append(
            {
                "run_id": run_id,
                "status": str(manifest.get("status", "")),
                "started_at": str(manifest.get("started_at", "")),
                "finished_at": str(manifest.get("finished_at", "")),
                "elapsed_seconds": manifest.get("elapsed_seconds", ""),
                "input_file": str(manifest.get("input_file", "")),
                "output_dir": output,
                "stage_count": result.get("stage_count", ""),
                "warning_count": result.get("warning_count", ""),
            }
        )
    rows.sort(key=lambda row: str(row["finished_at"] or row["started_at"]), reverse=True)
    return rows[:100]


def _case_manifests(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history_paths = [root / "case_history.jsonl", root / "runs" / "case_history.jsonl"]
    for history in history_paths:
        for row in _read_jsonl(history):
            if isinstance(row, dict):
                rows.append(row)
    for manifest_path in sorted((root / "runs").glob("*/case_manifest.json")):
        manifest = _read_json(manifest_path)
        if manifest:
            rows.append(dict(manifest))
    return rows


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for area in ARTIFACT_ROOTS:
        folder = root / area
        if not folder.exists():
            continue
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            if _excluded(path):
                continue
            stat = path.stat()
            rows.append(
                {
                    "path": _rel(root, path),
                    "role": _artifact_role(path),
                    "area": area,
                    "bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "run_id": _run_id_for_path(root, path),
                }
            )
    rows.sort(key=lambda row: (str(row["area"]), str(row["path"])))
    return rows


def _storage_summary(root: Path, *, artifact_rows: Sequence[Mapping[str, Any]], warning_bytes: int) -> dict[str, Any]:
    by_area: dict[str, dict[str, Any]] = {}
    for row in artifact_rows:
        area = str(row.get("area", "other"))
        bucket = by_area.setdefault(area, {"area": area, "bytes": 0, "file_count": 0, "warning": ""})
        bucket["bytes"] += int(row.get("bytes", 0) or 0)
        bucket["file_count"] += 1
    for area in PROJECT_DIRS:
        by_area.setdefault(area, {"area": area, "bytes": 0, "file_count": 0, "warning": ""})
    areas = sorted(by_area.values(), key=lambda item: str(item["area"]))
    total = sum(int(row["bytes"]) for row in areas)
    for row in areas:
        row["warning"] = "over_budget" if int(row["bytes"]) > warning_bytes else ""
    large_files = sorted((row for row in artifact_rows if int(row.get("bytes", 0) or 0) > warning_bytes), key=lambda row: int(row.get("bytes", 0) or 0), reverse=True)
    return {
        "total_bytes": total,
        "storage_warning_bytes": warning_bytes,
        "over_budget": total > warning_bytes,
        "areas": areas,
        "large_files": list(large_files[:20]),
    }


def _archive_plan(root: Path, storage: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for name in ("input", "runs", "reports", "results", "logs"):
        path = root / name
        if path.exists():
            candidates.append({"path": name, "kind": "directory"})
    for path in sorted(root.glob(f"*{PROJECT_EXTENSION}")):
        candidates.append({"path": path.name, "kind": "project_file"})
    return {
        "recommended_archive_name": f"{root.name or 'geofem'}_workspace_{datetime.now().strftime('%Y%m%d')}.zip",
        "candidate_roots": candidates,
        "total_bytes": int(storage.get("total_bytes", 0) or 0),
    }


def _iter_archive_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _excluded(path) or path.suffix.lower() in ARCHIVE_EXCLUDED_SUFFIXES:
            continue
        yield path


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _run_id_for_path(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return ""
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "runs":
        return parts[1]
    return ""


def _artifact_role(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gfemproj"):
        return "project"
    if name in {"case_manifest.json", "sample_project_manifest.json"}:
        return "manifest"
    if name == "summary.json":
        return "summary"
    if name.endswith(".csv"):
        return "table"
    if name.endswith(".html") or name.endswith(".pdf"):
        return "report"
    if name.endswith(".png"):
        return "image"
    if name.endswith((".yaml", ".yml", ".json")):
        return "configuration"
    return "artifact"


def _dashboard_html(dashboard: Mapping[str, Any]) -> str:
    counts = dashboard.get("counts", {}) if isinstance(dashboard.get("counts"), Mapping) else {}
    storage = dashboard.get("storage", {}) if isinstance(dashboard.get("storage"), Mapping) else {}
    project_rows = [[row.get("name", ""), row.get("path", ""), row.get("latest_run", ""), row.get("run_record_count", "")] for row in dashboard.get("projects", []) if isinstance(row, Mapping)]
    run_rows = [
        [row.get("run_id", ""), row.get("status", ""), row.get("finished_at", ""), row.get("elapsed_seconds", ""), row.get("output_dir", "")]
        for row in dashboard.get("runs", [])
        if isinstance(row, Mapping)
    ]
    area_rows = [[row.get("area", ""), row.get("bytes", ""), row.get("file_count", ""), row.get("warning", "")] for row in storage.get("areas", []) if isinstance(row, Mapping)]
    artifact_rows = [
        [row.get("path", ""), row.get("role", ""), row.get("area", ""), row.get("bytes", ""), row.get("modified_at", "")]
        for row in dashboard.get("artifacts", [])[:200]
        if isinstance(row, Mapping)
    ]
    summary_rows = [
        ["project_file_count", counts.get("project_file_count", 0)],
        ["recent_input_count", counts.get("recent_input_count", 0)],
        ["run_count", counts.get("run_count", 0)],
        ["artifact_count", counts.get("artifact_count", 0)],
        ["total_bytes", storage.get("total_bytes", 0)],
    ]
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>GeoFEM Workspace Dashboard</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>GeoFEM ワークスペースダッシュボード</h1>"
        f"<p>{html_escape(str(dashboard.get('project_root', '')))}</p>"
        "<h2>概要</h2>"
        f"{table(['項目', '値'], summary_rows)}"
        "<h2>プロジェクト</h2>"
        f"{table(['名前', 'ファイル', '最新解析', '解析記録数'], project_rows)}"
        "<h2>最近の解析</h2>"
        f"{table(['解析ID', '状態', '完了時刻', '経過秒', '出力先'], run_rows)}"
        "<h2>容量</h2>"
        f"{table(['領域', 'bytes', 'ファイル数', '警告'], area_rows)}"
        "<h2>成果物</h2>"
        f"{table(['パス', '種別', '領域', 'bytes', '更新時刻'], artifact_rows)}"
        "</body></html>"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


__all__ = [
    "WORKSPACE_ARCHIVE_SCHEMA",
    "WORKSPACE_DASHBOARD_SCHEMA",
    "archive_workspace",
    "build_workspace_dashboard",
    "write_workspace_dashboard",
]
