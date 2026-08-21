"""Startup environment diagnostics and support-package artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import getpass
import html
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import zipfile

from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


MAX_SUPPORT_FILE_BYTES = 2_000_000
DEFAULT_SUPPORT_EXCLUDE_PATTERNS = (
    "*.dmp",
    "*.dump",
    "*.bak",
    "*.tmp",
    "*.vtk",
    "*.vtu",
    "*.npy",
    "*.npz",
    "*.mp4",
    "*.avi",
    "*.zip",
)


@dataclass(frozen=True)
class SupportPackageOptions:
    """User-selectable support ZIP privacy and size policy."""

    exclude_personal_info: bool = True
    include_large_results: bool = False
    max_file_bytes: int = MAX_SUPPORT_FILE_BYTES
    exclude_patterns: tuple[str, ...] = DEFAULT_SUPPORT_EXCLUDE_PATTERNS


def write_startup_support_artifacts(
    report: Mapping[str, Any],
    output_dir: str | Path,
    report_paths: Mapping[str, str] | None = None,
    *,
    options: SupportPackageOptions | Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write first-run diagnostics, repair guidance, and a compact support ZIP."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    support_options = _support_options(options)
    package_path = out / "startup_support_package.zip"
    environment_payload = build_environment_diagnostics(report)
    repair_payload = build_repair_guide(report, support_package=package_path)

    paths = {
        "environment_json": str(out / "startup_environment_diagnostics.json"),
        "environment_csv": str(out / "startup_environment_diagnostics.csv"),
        "environment_html": str(out / "startup_environment_diagnostics.html"),
        "repair_guide_json": str(out / "startup_repair_guide.json"),
        "repair_guide_csv": str(out / "startup_repair_guide.csv"),
        "repair_guide_html": str(out / "startup_repair_guide.html"),
        "repair_guide_md": str(out / "startup_repair_guide.md"),
        "legacy_environment_json": str(out / "startup_environment.json"),
        "support_manifest": str(out / "startup_support_manifest.json"),
        "support_package": str(package_path),
    }

    write_json_artifact(paths["environment_json"], environment_payload)
    write_dict_rows_csv(
        paths["environment_csv"],
        environment_payload["diagnostics"],
        ["category", "name", "status", "detail", "path", "suggested_action"],
    )
    write_html_artifact(paths["environment_html"], _environment_html(environment_payload))

    write_json_artifact(paths["repair_guide_json"], repair_payload)
    write_dict_rows_csv(paths["repair_guide_csv"], repair_payload["actions"], ["priority", "status", "check", "action"])
    write_html_artifact(paths["repair_guide_html"], _repair_html(repair_payload))
    Path(paths["repair_guide_md"]).write_text(_repair_markdown(repair_payload), encoding="utf-8")
    write_json_artifact(paths["legacy_environment_json"], report.get("environment", {}))

    manifest = _support_manifest(report, paths)
    manifest["max_file_bytes"] = support_options.max_file_bytes
    manifest["support_package"] = _redact_text(paths["support_package"], support_options, report, out)
    manifest["privacy"] = _privacy_manifest(support_options)
    files = _support_files(report, out, paths, report_paths)
    included_files: list[tuple[Path, str]] = []
    excluded_large: list[dict[str, Any]] = []
    excluded_by_pattern: list[dict[str, Any]] = []
    for path in _unique_existing_files(files):
        size = path.stat().st_size
        arcname = path.name if path.parent == out else str(Path(path.parent.name) / path.name)
        if _excluded_by_pattern(path, arcname, support_options):
            excluded_by_pattern.append({"path": _redact_text(str(path), support_options, report, out), "archive_name": arcname, "reason": "matched support package exclude pattern"})
            continue
        if not support_options.include_large_results and size > support_options.max_file_bytes:
            excluded_large.append({"path": _redact_text(str(path), support_options, report, out), "archive_name": arcname, "bytes": size, "reason": "larger than support package limit"})
            continue
        included_files.append((path, arcname))
        manifest["included"].append({"path": _redact_text(str(path), support_options, report, out), "archive_name": arcname, "bytes": size})
    manifest["excluded_large"] = excluded_large
    manifest["excluded_large_count"] = len(excluded_large)
    manifest["excluded_by_pattern"] = excluded_by_pattern
    manifest["excluded_by_pattern_count"] = len(excluded_by_pattern)
    write_json_artifact(paths["support_manifest"], manifest)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in included_files:
            payload = _support_archive_bytes(path, support_options, report, out)
            if payload is None:
                zf.write(path, arcname=arcname)
            else:
                zf.writestr(arcname, payload)
        zf.write(paths["support_manifest"], arcname=Path(paths["support_manifest"]).name)
    return paths


def build_environment_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten startup checks into environment-focused diagnostic rows."""

    checks = [row for row in report.get("checks", []) if isinstance(row, Mapping)]
    diagnostics = [_diagnostic_row(row) for row in checks]
    environment = report.get("environment", {})
    features = [
        "python_runtime_probe",
        "cpu_and_architecture_probe",
        "gpu_tool_probe",
        "dependency_probe",
        "permission_probe",
        "numba_cache_probe",
        "gui_font_probe",
        "sample_analysis_probe",
        "version_license_inventory",
        "support_package_collection",
    ]
    return {
        "schema": "geofem.startup_environment_diagnostics.v1",
        "passed": bool(report.get("passed", False)),
        "error_count": int(report.get("error_count", 0) or 0),
        "warning_count": int(report.get("warning_count", 0) or 0),
        "features": features,
        "environment": environment,
        "diagnostics": diagnostics,
    }


def build_repair_guide(report: Mapping[str, Any], *, support_package: str | Path) -> dict[str, Any]:
    """Build a structured repair guide from failed and warning startup checks."""

    checks = [row for row in report.get("checks", []) if isinstance(row, Mapping)]
    actions: list[dict[str, Any]] = []
    for row in checks:
        status = str(row.get("status", ""))
        if status == "OK":
            continue
        name = str(row.get("name", ""))
        actions.append(
            {
                "priority": "high" if status == "ERROR" else "medium",
                "status": status,
                "check": name,
                "action": _suggested_action(name),
            }
        )
    if not actions:
        actions.append({"priority": "info", "status": "OK", "check": "startup", "action": "追加の修復作業は不要です。"})
    return {
        "schema": "geofem.startup_repair_guide.v1",
        "passed": bool(report.get("passed", False)),
        "error_count": int(report.get("error_count", 0) or 0),
        "warning_count": int(report.get("warning_count", 0) or 0),
        "support_package": str(support_package),
        "actions": actions,
        "checks": [_diagnostic_row(row) for row in checks],
    }


def _diagnostic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    name = str(row.get("name", ""))
    status = str(row.get("status", ""))
    return {
        "category": _category_for_name(name),
        "name": name,
        "status": status,
        "detail": str(row.get("detail", "")),
        "path": str(row.get("path", "")),
        "suggested_action": "" if status == "OK" else _suggested_action(name),
    }


def _category_for_name(name: str) -> str:
    if name.startswith("dependency:") or name.startswith("optional:"):
        return "dependency"
    if name.startswith("permission:"):
        return "permission"
    if name.startswith("environment:"):
        return "environment"
    if name.startswith("numba:"):
        return "acceleration"
    if name.startswith("font:"):
        return "font"
    if name.startswith("version_info:"):
        return "version_info"
    if name.startswith("file:"):
        return "distribution"
    if name.startswith("sample:"):
        return "sample"
    return "startup"


def _suggested_action(name: str) -> str:
    if name.startswith("dependency:") or name.startswith("optional:"):
        return "requirements.txt で Python 依存関係を再導入し、バージョンを再確認してください。"
    if name.startswith("file:"):
        return "配布補助ファイルの欠落を確認し、配布パッケージを再作成してください。"
    if name.startswith("permission:"):
        return "出力先、プロジェクト、ユーザー一時フォルダの読み書き権限を確認してください。"
    if name.startswith("numba:"):
        return "NUMBA_CACHE_DIR を書き込み可能なフォルダへ設定し、キャッシュ作成を再試行してください。"
    if name.startswith("font:"):
        return "Meiryo、Yu Gothic UI、Noto Sans CJK JP 相当の日本語フォントを利用可能にしてください。"
    if name.startswith("sample:"):
        return "サンプル解析の出力と startup_support_package.zip を添えて解析環境を確認してください。"
    if name.startswith("environment:gpu_probe"):
        return "GPU利用が必要な場合のみ、GPUドライバと nvidia-smi などのベンダーツールを確認してください。"
    return "startup_check.json と startup_repair_guide.html の内容を確認してください。"


def _support_manifest(report: Mapping[str, Any], paths: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": "geofem.startup_support_package.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": bool(report.get("passed", False)),
        "error_count": int(report.get("error_count", 0) or 0),
        "warning_count": int(report.get("warning_count", 0) or 0),
        "max_file_bytes": MAX_SUPPORT_FILE_BYTES,
        "support_package": paths["support_package"],
        "included": [],
    }


def _privacy_manifest(options: SupportPackageOptions) -> dict[str, Any]:
    data = asdict(options)
    data["features"] = [
        "personal_path_redaction",
        "large_result_exclusion",
        "user_exclude_patterns",
        "redacted_archive_payloads",
    ]
    return data


def _support_files(
    report: Mapping[str, Any],
    out: Path,
    paths: Mapping[str, str],
    report_paths: Mapping[str, str] | None,
) -> list[Path]:
    files = [Path(value) for key, value in paths.items() if key != "support_package"]
    for value in (report_paths or {}).values():
        files.append(Path(value))
    sample = report.get("sample_output")
    if sample:
        sample_dir = Path(str(sample))
        for name in ("summary.json", "result_view_index.json", "standard_report.html"):
            files.append(sample_dir / name)
    version_info = report.get("version_info", {})
    if isinstance(version_info, Mapping):
        artifacts = version_info.get("artifacts", {})
        if isinstance(artifacts, Mapping):
            for value in artifacts.values():
                files.append(Path(str(value)))
    audit = Path(str(report.get("project_root", ""))) / ".geofem_audit_log.jsonl"
    if audit.exists():
        files.append(audit)
    return files


def _unique_existing_files(files: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in files:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _support_options(value: SupportPackageOptions | Mapping[str, Any] | None) -> SupportPackageOptions:
    if isinstance(value, SupportPackageOptions):
        return value
    if isinstance(value, Mapping):
        patterns = value.get("exclude_patterns", DEFAULT_SUPPORT_EXCLUDE_PATTERNS)
        if isinstance(patterns, str):
            pattern_tuple = tuple(part.strip() for part in patterns.split(";") if part.strip())
        elif isinstance(patterns, Sequence):
            pattern_tuple = tuple(str(part) for part in patterns)
        else:
            pattern_tuple = DEFAULT_SUPPORT_EXCLUDE_PATTERNS
        return SupportPackageOptions(
            exclude_personal_info=bool(value.get("exclude_personal_info", True)),
            include_large_results=bool(value.get("include_large_results", False)),
            max_file_bytes=max(1, int(value.get("max_file_bytes", MAX_SUPPORT_FILE_BYTES) or MAX_SUPPORT_FILE_BYTES)),
            exclude_patterns=pattern_tuple or DEFAULT_SUPPORT_EXCLUDE_PATTERNS,
        )
    return SupportPackageOptions()


def _excluded_by_pattern(path: Path, arcname: str, options: SupportPackageOptions) -> bool:
    candidates = {path.name, arcname, str(path)}
    return any(fnmatch.fnmatch(candidate.lower(), pattern.lower()) for candidate in candidates for pattern in options.exclude_patterns)


def _support_archive_bytes(path: Path, options: SupportPackageOptions, report: Mapping[str, Any], out: Path) -> bytes | None:
    if not options.exclude_personal_info:
        return None
    if path.suffix.lower() not in {".json", ".csv", ".html", ".htm", ".md", ".txt", ".log", ".jsonl", ".yaml", ".yml"}:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except Exception:
            return None
    except Exception:
        return None
    return _redact_text(text, options, report, out).encode("utf-8")


def _redact_text(text: str, options: SupportPackageOptions, report: Mapping[str, Any], out: Path) -> str:
    if not options.exclude_personal_info:
        return text
    replacements: list[tuple[str, str]] = []
    for label, value in (
        ("<PROJECT_ROOT>", report.get("project_root", "")),
        ("<OUTPUT_DIR>", out),
        ("<HOME>", Path.home()),
        ("<CWD>", Path.cwd()),
    ):
        if value:
            replacements.append((str(value), label))
            try:
                replacements.append((str(Path(str(value)).resolve()), label))
            except Exception:
                pass
    user = getpass.getuser()
    if user:
        replacements.append((user, "<USER>"))
    redacted = str(text)
    for source, target in sorted(set(replacements), key=lambda item: len(item[0]), reverse=True):
        if source:
            redacted = redacted.replace(source, target)
            redacted = redacted.replace(source.replace("\\", "/"), target)
    return redacted


def _environment_html(payload: Mapping[str, Any]) -> str:
    rows = [
        [
            row.get("category", ""),
            row.get("name", ""),
            row.get("status", ""),
            row.get("detail", ""),
            row.get("path", ""),
            row.get("suggested_action", ""),
        ]
        for row in payload.get("diagnostics", [])
        if isinstance(row, Mapping)
    ]
    return html_table_document(
        title="GeoFEM 起動環境診断",
        lead=f"errors={payload.get('error_count', 0)}, warnings={payload.get('warning_count', 0)}",
        headers=["category", "name", "status", "detail", "path", "suggested_action"],
        rows=rows,
    )


def _repair_html(payload: Mapping[str, Any]) -> str:
    action_rows = [
        [row.get("priority", ""), row.get("status", ""), row.get("check", ""), row.get("action", "")]
        for row in payload.get("actions", [])
        if isinstance(row, Mapping)
    ]
    support = html.escape(str(payload.get("support_package", "")))
    return html_table_document(
        title="GeoFEM 起動診断 修復ガイド",
        lead=f"support package: {support}",
        headers=["priority", "status", "check", "action"],
        rows=action_rows,
    )


def _repair_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# GeoFEM 起動診断 修復ガイド",
        "",
        f"- passed: `{bool(payload.get('passed', False))}`",
        f"- errors: `{int(payload.get('error_count', 0) or 0)}`",
        f"- warnings: `{int(payload.get('warning_count', 0) or 0)}`",
        f"- support package: `{payload.get('support_package', '')}`",
        "",
        "## 推奨対応",
        "",
    ]
    for row in payload.get("actions", []):
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('status', '')}` {row.get('check', '')}: {row.get('action', '')}")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_SUPPORT_EXCLUDE_PATTERNS",
    "MAX_SUPPORT_FILE_BYTES",
    "SupportPackageOptions",
    "build_environment_diagnostics",
    "build_repair_guide",
    "write_startup_support_artifacts",
]
