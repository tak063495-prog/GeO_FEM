"""Distribution and first-run startup verification for GeoFEM."""

from __future__ import annotations

import csv
import html
import importlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

from .fem2d_solver import solve_plane_strain_config
from .messages import message
from .samples import plane_strain_quad4_sample
from .startup_support import SupportPackageOptions, write_startup_support_artifacts
from .version_info import build_version_info, write_version_info_artifacts
from .gui.font_support import PREFERRED_GUI_FONTS, preferred_gui_font_inventory


REQUIRED_MODULES = ("numpy", "scipy", "numba", "yaml")
OPTIONAL_MODULES = ("PySide6", "PyInstaller")
REQUIRED_FILES = ("run_gui.bat", "build_gui_exe.bat", "requirements.txt")


def run_startup_check(
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    include_gui: bool = False,
    run_sample: bool = True,
    support_options: SupportPackageOptions | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a Windows-ready distribution can import, start, and solve a sample."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str = "", path: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail, "path": path})

    add("python", "OK", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} on {platform.platform()}")
    environment = _environment_summary(root, out)
    add("environment:python_executable", "OK", environment["python_executable"])
    add("environment:cpu", "OK", f"{environment['processor'] or 'unknown'}, cores={environment['cpu_count']}")
    add("environment:architecture", "OK", f"{environment['machine']} / {environment['python_architecture']}")
    add("environment:gpu_probe", "OK" if environment["gpu_probe"]["available"] else "WARN", environment["gpu_probe"]["detail"])
    add("permission:output_dir_write", "OK" if _can_write_directory(out) else "ERROR", "startup-check output directory is writable", str(out))
    add("permission:project_root_read", "OK" if root.exists() and os.access(root, os.R_OK) else "ERROR", "project root is readable", str(root))
    for module_name in REQUIRED_MODULES:
        version = _module_version(module_name)
        add(f"dependency:{module_name}", "OK" if _can_import(module_name) else "ERROR", f"required Python module, version={version}")
    for module_name in OPTIONAL_MODULES:
        status = "OK" if _can_import(module_name) else ("WARN" if module_name == "PyInstaller" else "WARN")
        detail = f"optional GUI/packaging module, version={_module_version(module_name)}"
        if module_name == "PySide6" and include_gui and status != "OK":
            status = "ERROR"
            detail = "required because include_gui=True"
        add(f"optional:{module_name}", status, detail)
    numba_cache = _numba_cache_probe(out)
    add("numba:cache_write", "OK" if numba_cache["writable"] else "WARN", numba_cache["detail"], numba_cache["path"])
    font_probe = _font_probe(include_gui=include_gui)
    add("font:jp_gui", font_probe["status"], font_probe["detail"])
    for file_name in REQUIRED_FILES:
        path = root / file_name
        add(f"file:{file_name}", "OK" if path.exists() else "ERROR", "distribution helper file", str(path))

    version_payload = build_version_info(project_root=root, include_gui=include_gui)
    version_paths = write_version_info_artifacts(out, project_root=root, include_gui=include_gui, payload=version_payload)
    dep_summary = version_payload.get("dependency_summary", {})
    font_summary = version_payload.get("font_summary", {})
    external_summary = version_payload.get("external_tool_summary", {})
    add(
        "version_info:dependency_licenses",
        "OK" if int(dep_summary.get("missing_required_count", 0) or 0) == 0 else "ERROR",
        f"dependencies={dep_summary.get('dependency_count', 0)}, unknown_licenses={dep_summary.get('unknown_license_count', 0)}",
        version_paths["json"],
    )
    add(
        "version_info:external_tools",
        "OK",
        f"external tools={external_summary.get('tool_count', 0)}, available={external_summary.get('available_count', 0)}",
        version_paths["html"],
    )
    add(
        "version_info:fonts",
        "OK" if not include_gui or str(font_summary.get("status", "")) == "available" else "WARN",
        f"font probe status={font_summary.get('status', 'skipped')}",
        version_paths["csv"],
    )

    sample_output = None
    elapsed = 0.0
    if run_sample:
        sample_dir = out / "sample_run"
        t0 = time.perf_counter()
        try:
            result = solve_plane_strain_config(plane_strain_quad4_sample(integration="B-bar"), sample_dir)
            elapsed = time.perf_counter() - t0
            sample_output = str(result.output_dir)
            add("sample:solve", "OK", f"stages={len(result.stages)}, elapsed={elapsed:.3f}s", sample_output)
            for required in ("summary.json", "result_view_index.json", "standard_report.html"):
                path = result.output_dir / required
                add(f"sample:artifact:{required}", "OK" if path.exists() else "ERROR", "sample analysis artifact", str(path))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            add("sample:solve", "ERROR", str(exc), str(sample_dir))

    error_count = sum(1 for row in checks if row["status"] == "ERROR")
    warning_count = sum(1 for row in checks if row["status"] == "WARN")
    report = {
        "schema": "geofem.startup_check.v1",
        "passed": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "project_root": str(root),
        "sample_output": sample_output,
        "sample_elapsed_seconds": elapsed,
        "environment": environment,
        "version_info": {"summary": version_payload, "artifacts": version_paths},
        "repair_actions": _repair_actions(checks),
        "checks": checks,
    }
    report["support_package"] = str(out / "startup_support_package.zip")
    paths = write_startup_check_report(report, out)
    support_artifacts = write_startup_support_artifacts(report, out, paths, options=support_options)
    report["support_package"] = support_artifacts["support_package"]
    report["support_artifacts"] = support_artifacts
    merged_paths = dict(paths)
    merged_paths.update(support_artifacts)
    report["artifacts"] = merged_paths
    paths = write_startup_check_report(report, out)
    report["artifacts"].update(paths)
    return report


def write_startup_check_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "startup_check.json"
    csv_path = out / "startup_check.csv"
    html_path = out / "startup_check.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "status", "detail", "path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report.get("checks", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fields})
    html_path.write_text(_startup_html(report), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def write_startup_support_package(
    report: Mapping[str, Any],
    output_dir: str | Path,
    report_paths: Mapping[str, str] | None = None,
    *,
    options: SupportPackageOptions | Mapping[str, Any] | None = None,
) -> str:
    """Write repair guidance and a compact support ZIP for first-run issues."""

    return write_startup_support_artifacts(report, output_dir, report_paths, options=options)["support_package"]


def _startup_html(report: Mapping[str, Any]) -> str:
    rows = []
    for row in report.get("checks", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('detail', '')))}</td>"
            f"<td>{html.escape(str(row.get('path', '')))}</td>"
            "</tr>"
        )
    heading = message("startup.heading")
    repair_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in report.get("repair_actions", []))
    support = html.escape(str(report.get("support_package", "")))
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{html.escape(heading)}</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>{html.escape(heading)}</h1>
<p>passed={bool(report.get('passed', False))}, errors={int(report.get('error_count', 0) or 0)}, warnings={int(report.get('warning_count', 0) or 0)}</p>
<p>support package: {support}</p>
<h2>Repair guide</h2><ul>{repair_rows or '<li>No immediate repair action is required.</li>'}</ul>
<table><thead><tr><th>name</th><th>status</th><th>detail</th><th>path</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def _environment_summary(project_root: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_architecture": platform.architecture()[0],
        "cpu_count": os.cpu_count() or 0,
        "working_directory": str(Path.cwd()),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "numba_cache_dir": str(os.environ.get("NUMBA_CACHE_DIR", "")),
        "path_entries": len(os.environ.get("PATH", "").split(os.pathsep)),
        "gpu_probe": _gpu_probe(),
    }


def _gpu_probe() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        return {"available": True, "detail": f"nvidia-smi found: {nvidia_smi}", "tool": nvidia_smi}
    return {"available": False, "detail": "No GPU probe tool was found; CPU execution remains supported.", "tool": ""}


def _can_write_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".geofem_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _numba_cache_probe(output_dir: Path) -> dict[str, Any]:
    try:
        import numba  # type: ignore

        cache_root = Path(os.environ.get("NUMBA_CACHE_DIR") or output_dir / "numba_cache_probe")
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / ".write_probe"
        probe.write_text(str(getattr(numba, "__version__", "unknown")), encoding="utf-8")
        probe.unlink()
        return {"writable": True, "path": str(cache_root), "detail": f"Numba {getattr(numba, '__version__', 'unknown')} cache path is writable."}
    except Exception as exc:
        return {"writable": False, "path": str(output_dir / "numba_cache_probe"), "detail": f"Numba cache probe failed: {exc}"}


def _font_probe(*, include_gui: bool) -> dict[str, str]:
    if not include_gui:
        return {"status": "OK", "detail": "Skipped detailed GUI font probe because include_gui=False."}
    try:
        from PySide6.QtWidgets import QApplication  # type: ignore
        from PySide6.QtGui import QFontDatabase  # type: ignore

        app = QApplication.instance()
        created = False
        if app is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            app = QApplication([])
            created = True
        inventory = preferred_gui_font_inventory(QFontDatabase)
        available = list(inventory["available_preferred"])
        if created:
            app.quit()
        if available:
            loaded = ", ".join(str(name) for name in inventory.get("loaded_families", []))
            suffix = f" (registered: {loaded})" if loaded else ""
            return {"status": "OK", "detail": "Japanese GUI font available: " + ", ".join(available) + suffix}
        return {"status": "WARN", "detail": "No preferred Japanese GUI font was detected. Preferred: " + ", ".join(PREFERRED_GUI_FONTS)}
    except Exception as exc:
        return {"status": "WARN", "detail": f"GUI font probe failed: {exc}"}


def _repair_actions(checks: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in checks:
        name = str(row.get("name", ""))
        status = str(row.get("status", ""))
        if status == "OK":
            continue
        if name.startswith("dependency:") or name.startswith("optional:"):
            actions.append("Python依存関係を再導入してください: python -m pip install -r requirements.txt")
        elif name.startswith("file:"):
            actions.append("配布補助ファイルが不足しています。リポジトリ直下の run_gui.bat / build_gui_exe.bat / requirements.txt を確認してください。")
        elif name.startswith("permission:"):
            actions.append("出力先またはプロジェクトフォルダの読み書き権限を確認し、書き込み可能な runs フォルダを指定してください。")
        elif name.startswith("numba:"):
            actions.append("NUMBA_CACHE_DIR を書き込み可能なフォルダへ設定するか、ユーザーキャッシュ権限を確認してください。")
        elif name.startswith("font:"):
            actions.append("Windows日本語フォントまたは Meiryo / Yu Gothic UI 相当のフォントを導入してください。")
        elif name.startswith("sample:"):
            actions.append("サンプル解析ログと startup_support_package.zip を添えて解析環境を確認してください。")
        elif name.startswith("environment:gpu_probe"):
            actions.append("GPUは任意です。GPU利用が必要な場合はベンダーツールとドライバを確認してください。")
    return sorted(set(actions))


def _repair_guide_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# GeoFEM 起動診断 修復ガイド",
        "",
        f"- passed: `{bool(report.get('passed', False))}`",
        f"- errors: `{int(report.get('error_count', 0) or 0)}`",
        f"- warnings: `{int(report.get('warning_count', 0) or 0)}`",
        "",
        "## 推奨対応",
        "",
    ]
    actions = [str(item) for item in report.get("repair_actions", [])]
    if actions:
        lines.extend(f"- {item}" for item in actions)
    else:
        lines.append("- 追加の修復作業は不要です。")
    lines.extend(["", "## 主要診断", ""])
    for row in report.get("checks", []):
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('status', '')}` {row.get('name', '')}: {row.get('detail', '')}")
    return "\n".join(lines) + "\n"


def _unique_existing_files(files: list[Path]) -> list[Path]:
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


def _module_version(module_name: str) -> str:
    package_name = "PyYAML" if module_name == "yaml" else module_name
    try:
        return metadata.version(package_name)
    except Exception:
        try:
            module = importlib.import_module(module_name)
            return str(getattr(module, "__version__", "unknown"))
        except Exception:
            return "not installed"


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


__all__ = [
    "REQUIRED_FILES",
    "REQUIRED_MODULES",
    "OPTIONAL_MODULES",
    "run_startup_check",
    "write_startup_check_report",
    "write_startup_support_package",
]
