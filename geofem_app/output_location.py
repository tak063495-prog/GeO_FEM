"""Output directory resolution and run artifact manifests."""

from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any, Mapping


_TOKEN_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _safe_path_part(value: Any, *, fallback: str = "analysis") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text or fallback


def _render_run_folder(template: Any, *, input_stem: str, timestamp: str) -> str:
    raw = str(template or "{input_stem}_run_{timestamp}")
    values = {
        "input_stem": _safe_path_part(input_stem),
        "timestamp": _safe_path_part(timestamp, fallback="run"),
    }

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return values.get(key, match.group(0))

    rendered = _TOKEN_PATTERN.sub(replace, raw)
    return _safe_path_part(rendered, fallback=f"{values['input_stem']}_run_{values['timestamp']}")


def _as_output_mapping(output_config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(output_config) if isinstance(output_config, Mapping) else {}


def resolve_analysis_output_dir(
    config_path: str | Path | None,
    output_config: Mapping[str, Any] | None,
    project_root: str | Path,
    explicit_out: str | Path | None = None,
    timestamp: str | datetime | None = None,
    *,
    default_root_policy: str = "project_runs",
) -> Path:
    """Resolve a run output directory for CLI and GUI analysis execution.

    Compatibility rule: a legacy ``output.directory`` without ``root_policy``
    remains the exact output directory. New root policies use ``run_folder``
    below the selected root.
    """

    project = Path(project_root).resolve()
    cfg_path = Path(config_path).resolve() if config_path is not None else None
    input_parent = cfg_path.parent if cfg_path is not None else project
    input_stem = cfg_path.stem if cfg_path is not None else "analysis"
    if explicit_out:
        return Path(explicit_out).expanduser().resolve()

    output = _as_output_mapping(output_config)
    policy = str(output.get("root_policy", output.get("root", default_root_policy)) or default_root_policy).strip().lower()
    directory_value = output.get("directory", output.get("dir", ""))

    if policy in {"", "legacy"} and directory_value:
        policy = "legacy_directory"

    if not output.get("root_policy") and directory_value:
        path = Path(str(directory_value)).expanduser()
        return (path if path.is_absolute() else input_parent / path).resolve()

    if isinstance(timestamp, datetime):
        timestamp_text = timestamp.strftime("%Y%m%d_%H%M%S")
    elif timestamp is not None:
        timestamp_text = str(timestamp)
    else:
        timestamp_text = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = _render_run_folder(output.get("run_folder"), input_stem=input_stem, timestamp=timestamp_text)

    if policy in {"same_as_input", "input", "input_dir", "input-directory"}:
        root = input_parent
    elif policy in {"custom", "custom_directory", "custom-dir"}:
        if directory_value:
            custom = Path(str(directory_value)).expanduser()
            root = custom if custom.is_absolute() else input_parent / custom
        else:
            root = project / "runs"
    else:
        root = project / "runs"
    return (root / run_folder).resolve()


def collect_run_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Collect standard result artifacts for a lightweight run manifest."""

    out = Path(output_dir)
    files = {
        "summary": out / "summary.json",
        "run_log": out / "run.log",
        "gui_log": out / "gui.log",
        "failure_report": out / "failure_report.json",
    }
    csv_files = sorted(out.rglob("*.csv")) if out.exists() else []
    vtk_files = sorted([*out.rglob("*.vtk"), *out.rglob("*.vtu")]) if out.exists() else []
    html_files = sorted(out.rglob("*.html")) if out.exists() else []
    return {
        "files": {name: str(path) for name, path in files.items() if path.exists()},
        "csv": [str(path) for path in csv_files],
        "vtk": [str(path) for path in vtk_files],
        "html": [str(path) for path in html_files],
    }


def write_run_manifest(
    *,
    output_dir: str | Path,
    input_path: str | Path | None,
    status: str,
    started_at: str,
    elapsed_seconds: float | None = None,
    output_config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a concise manifest beside ``summary.json`` and ``run.log``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "geofem.run_manifest.v1",
        "status": str(status),
        "started_at": str(started_at),
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
        "input_file": str(Path(input_path)) if input_path is not None else "",
        "output_dir": str(out),
        "output": dict(output_config or {}) if isinstance(output_config, Mapping) else {},
        "artifacts": collect_run_artifacts(out),
    }
    if extra:
        payload["extra"] = dict(extra)
    path = out / "run_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


__all__ = ["collect_run_artifacts", "resolve_analysis_output_dir", "write_run_manifest"]
