"""UTF-8 console and text-artifact audit helpers for GeoFEM."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "runs",
    "venv",
}

MOJIBAKE_MARKERS = (
    "\ufffd",
    "\u7e3a",
    "\u8b41",
    "\u8373",
    "\u870a",
    "\u879f",
    "\u90b1",
)


def configure_utf8_console() -> dict[str, str]:
    """Prefer UTF-8 for CLI output while keeping redirected streams usable."""

    result: dict[str, str] = {}
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        result[name] = str(getattr(stream, "encoding", "") or "")
        result[f"{name}_errors"] = str(getattr(stream, "errors", "") or "")
    return result


def audit_text_encoding(
    root: str | Path,
    *,
    suffixes: Iterable[str] = TEXT_SUFFIXES,
    excluded_dir_names: Iterable[str] = EXCLUDED_DIR_NAMES,
    max_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Scan text artifacts for UTF-8 decode failures and common mojibake markers."""

    root_path = Path(root)
    suffix_set = {item.lower() for item in suffixes}
    excluded = {item.lower() for item in excluded_dir_names}
    rows: list[dict[str, Any]] = []
    for path in _candidate_files(root_path, suffix_set, excluded):
        rel = _relative(path, root_path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            rows.append(_row(rel, "stat", "ERROR", False, str(exc)))
            continue
        if size > max_bytes:
            rows.append(_row(rel, "size", "WARN", False, f"skipped large text candidate ({size} bytes)"))
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            rows.append(_row(rel, "read", "ERROR", False, str(exc)))
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            rows.append(_row(rel, "utf8_decode", "ERROR", False, str(exc), size=size))
            continue
        marker_hits = _mojibake_hits(text)
        if marker_hits:
            rows.append(_row(rel, "mojibake_marker", "ERROR", False, ",".join(marker_hits), size=size))
        elif text.startswith("\ufeff"):
            rows.append(_row(rel, "utf8_bom", "WARN", False, "UTF-8 BOM is present; plain UTF-8 without BOM is preferred.", size=size))
        else:
            rows.append(_row(rel, "utf8_text", "INFO", True, "UTF-8 text file is readable.", size=size))

    errors = [row for row in rows if row["severity"] == "ERROR" and not row["passed"]]
    warnings = [row for row in rows if row["severity"] == "WARN" and not row["passed"]]
    return {
        "schema": "geofem.encoding_policy.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root_path),
        "passed": not errors,
        "file_count": len(rows),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "features": [
            "utf8_decode_audit",
            "mojibake_marker_detection",
            "utf8_console_reconfiguration",
            "text_artifact_encoding_policy",
        ],
        "checks": rows,
    }


def write_encoding_audit(
    root: str | Path,
    output_dir: str | Path,
    *,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    """Write JSON/CSV/HTML encoding audit artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = audit_text_encoding(root)
    summary["fail_on_warning"] = bool(fail_on_warning)
    summary["passed_with_warning_policy"] = bool(summary["passed"] and (not fail_on_warning or int(summary["warning_count"]) == 0))
    json_path = out / "encoding_audit.json"
    csv_path = out / "encoding_audit.csv"
    html_path = out / "encoding_audit.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(csv_path, summary["checks"])
    html_path.write_text(_html(summary), encoding="utf-8")
    summary["paths"] = {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}
    return summary


def _candidate_files(root: Path, suffixes: set[str], excluded: set[str]) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    paths: list[Path] = []
    for path in root.rglob("*"):
        if any(part.lower() in excluded for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            paths.append(path)
    return sorted(paths, key=lambda item: str(item).lower())


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mojibake_hits(text: str) -> list[str]:
    return [marker for marker in MOJIBAKE_MARKERS if marker in text]


def _row(path: str, check: str, severity: str, passed: bool, detail: str, *, size: int | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "check": check,
        "severity": severity,
        "passed": bool(passed),
        "size_bytes": "" if size is None else int(size),
        "detail": detail,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = ["path", "check", "severity", "passed", "size_bytes", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _html(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('path', '')))}</td>"
        f"<td>{html.escape(str(row.get('check', '')))}</td>"
        f"<td>{html.escape(str(row.get('severity', '')))}</td>"
        f"<td>{html.escape(str(row.get('passed', '')))}</td>"
        f"<td>{html.escape(str(row.get('detail', '')))}</td>"
        "</tr>"
        for row in summary.get("checks", [])
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>GeoFEM encoding audit</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px;text-align:left;}th{background:#f2f2f2;}</style></head><body>"
        "<h1>GeoFEM encoding audit</h1>"
        f"<p>passed={html.escape(str(summary.get('passed')))} errors={html.escape(str(summary.get('error_count')))} "
        f"warnings={html.escape(str(summary.get('warning_count')))}</p>"
        "<table><thead><tr><th>path</th><th>check</th><th>severity</th><th>passed</th><th>detail</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )


__all__ = [
    "EXCLUDED_DIR_NAMES",
    "MOJIBAKE_MARKERS",
    "TEXT_SUFFIXES",
    "audit_text_encoding",
    "configure_utf8_console",
    "write_encoding_audit",
]
