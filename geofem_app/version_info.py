"""Version, dependency, license, and external-tool inventory for GeoFEM."""

from __future__ import annotations

import csv
import html
import importlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Mapping

from . import __version__
from .cad_dwg_converter import dwg_converter_candidates, discover_dwg_converter
from .gui.font_support import PREFERRED_GUI_FONTS, preferred_gui_font_inventory
from .update_compatibility import CONFIG_SCHEMA_VERSION


PYTHON_DEPENDENCIES: tuple[dict[str, Any], ...] = (
    {"package": "numpy", "module": "numpy", "required": True, "purpose": "array kernels and numerical data"},
    {"package": "scipy", "module": "scipy", "required": True, "purpose": "sparse solvers and numerical utilities"},
    {"package": "numba", "module": "numba", "required": True, "purpose": "JIT-accelerated FEM and seepage kernels"},
    {"package": "PyYAML", "module": "yaml", "required": True, "purpose": "YAML input/output"},
    {"package": "PySide6", "module": "PySide6", "required": False, "purpose": "desktop GUI"},
    {"package": "pyinstaller", "module": "PyInstaller", "required": False, "purpose": "optional Windows executable packaging"},
)

def build_version_info(
    *,
    project_root: str | Path | None = None,
    include_gui: bool = False,
) -> dict[str, Any]:
    """Build the payload shown in the GUI about/version screen."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    dependencies = [_dependency_row(spec) for spec in PYTHON_DEPENDENCIES]
    external_tools = _external_tool_rows()
    font_inventory = _font_inventory(include_gui=include_gui)
    missing_required = [row for row in dependencies if row["required"] and row["status"] != "installed"]
    unknown_licenses = [row for row in dependencies if row["status"] == "installed" and not row["license"]]
    warning_count = len(unknown_licenses)
    return {
        "schema": "geofem.version_info.v1",
        "product": {
            "name": "GeoFEM 2D",
            "version": __version__,
            "input_config_schema": CONFIG_SCHEMA_VERSION,
            "project_root": str(root),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependency_summary": {
            "dependency_count": len(dependencies),
            "installed_count": sum(1 for row in dependencies if row["status"] == "installed"),
            "missing_required_count": len(missing_required),
            "unknown_license_count": len(unknown_licenses),
        },
        "external_tool_summary": {
            "tool_count": len(external_tools),
            "configured_count": sum(1 for row in external_tools if row["status"] == "configured"),
            "available_count": sum(1 for row in external_tools if row["status"] in {"configured", "detected"}),
        },
        "font_summary": {
            "checked": bool(font_inventory.get("checked", False)),
            "available_preferred_count": len(font_inventory.get("available_preferred", [])),
            "status": font_inventory.get("status", "skipped"),
        },
        "features": [
            "dependency_license_inventory",
            "external_tool_detection",
            "gui_font_inventory",
            "version_about_payload",
            "distribution_notice",
            "update_compatibility_policy",
        ],
        "distribution_notice": {
            "native_dwg_decode": "not_implemented",
            "dwg_policy": "DWG input is accepted through an external converter to DXF/SXF; native binary parity is not claimed.",
            "font_policy": "The GUI uses available system fonts from the preferred Japanese/Latin font list.",
            "license_policy": "Package license fields are collected from installed Python distribution metadata.",
            "update_policy": "Use the update-compatibility artifacts to migrate input schemas and decide whether generated results must be revalidated.",
            "input_config_schema": CONFIG_SCHEMA_VERSION,
        },
        "passed": not missing_required,
        "error_count": len(missing_required),
        "warning_count": warning_count,
        "dependencies": dependencies,
        "external_tools": external_tools,
        "fonts": font_inventory,
    }


def write_version_info_artifacts(
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    include_gui: bool = False,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write JSON/CSV/HTML artifacts for the version/about screen."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = dict(payload or build_version_info(project_root=project_root, include_gui=include_gui))
    json_path = out / "version_info.json"
    csv_path = out / "version_info.csv"
    html_path = out / "version_info.html"
    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(info, csv_path)
    html_path.write_text(_html(info), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _dependency_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    package = str(spec.get("package", ""))
    module_name = str(spec.get("module", package))
    required = bool(spec.get("required", False))
    installed = _can_import(module_name)
    version = _metadata_version(package, module_name) if installed else ""
    meta = _metadata(package) if installed else {}
    license_text, license_source = _license_from_metadata(meta)
    classifiers = [item for item in meta.get_all("Classifier", []) if "License" in str(item)] if hasattr(meta, "get_all") else []
    return {
        "category": "dependency",
        "name": package,
        "module": module_name,
        "required": required,
        "status": "installed" if installed else ("missing_required" if required else "missing_optional"),
        "version": version,
        "license": license_text,
        "license_source": license_source,
        "license_classifiers": "; ".join(str(item) for item in classifiers),
        "summary": str(meta.get("Summary", "")) if isinstance(meta, Mapping) else "",
        "homepage": str(meta.get("Home-page", "")) if isinstance(meta, Mapping) else "",
        "purpose": str(spec.get("purpose", "")),
        "path": _module_path(module_name) if installed else "",
    }


def _external_tool_rows() -> list[dict[str, Any]]:
    configured = os.environ.get("GEOFEM_DWG_CONVERTER", "")
    detected = discover_dwg_converter()
    if configured:
        status = "configured"
        path = configured
    elif detected:
        status = "detected"
        path = str(detected)
    else:
        status = "not_configured"
        path = ""
    return [
        {
            "category": "external_tool",
            "name": "DWG converter",
            "status": status,
            "version": "",
            "license": "managed outside GeoFEM",
            "license_source": "user-installed external tool",
            "purpose": "optional DWG to DXF/SXF conversion",
            "path": path,
            "candidates": "; ".join(dwg_converter_candidates()),
            "detail": "Native DWG binary decoding is not implemented; conversion through an external tool is required.",
        }
    ]


def _font_inventory(*, include_gui: bool) -> dict[str, Any]:
    if not include_gui:
        return {
            "checked": False,
            "status": "skipped",
            "preferred": list(PREFERRED_GUI_FONTS),
            "available_preferred": [],
            "detail": "GUI font probing is skipped unless include_gui=True.",
        }
    try:
        from PySide6.QtWidgets import QApplication  # type: ignore
        from PySide6.QtGui import QFontDatabase  # type: ignore

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance()
        created = False
        if app is None:
            app = QApplication([])
            created = True
        inventory = preferred_gui_font_inventory(QFontDatabase)
        available = list(inventory["available_preferred"])
        if created:
            app.quit()
        return {
            "checked": True,
            "status": "available" if available else "missing_preferred",
            "preferred": list(PREFERRED_GUI_FONTS),
            "available_preferred": available,
            "loaded_families": inventory.get("loaded_families", []),
            "registered_known_font_count": inventory.get("registered_known_font_count", 0),
            "detail": "Preferred GUI font probe completed.",
        }
    except Exception as exc:
        return {
            "checked": True,
            "status": "probe_failed",
            "preferred": list(PREFERRED_GUI_FONTS),
            "available_preferred": [],
            "detail": str(exc),
        }


def _write_csv(info: Mapping[str, Any], path: Path) -> None:
    fields = ["category", "name", "status", "required", "version", "license", "license_source", "purpose", "path", "detail"]
    rows: list[dict[str, Any]] = []
    for row in info.get("dependencies", []):
        if isinstance(row, Mapping):
            rows.append({field: row.get(field, "") for field in fields})
    for row in info.get("external_tools", []):
        if isinstance(row, Mapping):
            rows.append({field: row.get(field, "") for field in fields})
    fonts = info.get("fonts", {})
    if isinstance(fonts, Mapping):
        rows.append(
            {
                "category": "font",
                "name": "preferred GUI fonts",
                "status": fonts.get("status", ""),
                "required": False,
                "version": "",
                "license": "system font",
                "license_source": "operating system or user installation",
                "purpose": "GUI text rendering",
                "path": "",
                "detail": "; ".join(str(item) for item in fonts.get("available_preferred", [])) or fonts.get("detail", ""),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html(info: Mapping[str, Any]) -> str:
    product = info.get("product", {}) if isinstance(info.get("product", {}), Mapping) else {}
    rows = []
    for row in list(info.get("dependencies", [])) + list(info.get("external_tools", [])):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('category', 'dependency')))}</td>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('required', '')))}</td>"
            f"<td>{html.escape(str(row.get('version', '')))}</td>"
            f"<td>{html.escape(str(row.get('license', '')))}</td>"
            f"<td>{html.escape(str(row.get('purpose', '')))}</td>"
            f"<td>{html.escape(str(row.get('path', '')))}</td>"
            "</tr>"
        )
    fonts = info.get("fonts", {}) if isinstance(info.get("fonts", {}), Mapping) else {}
    notice = info.get("distribution_notice", {}) if isinstance(info.get("distribution_notice", {}), Mapping) else {}
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM Version Information</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ccd2d8;padding:6px;text-align:left;vertical-align:top}}th{{background:#eef2f7}}</style></head>
<body>
<h1>GeoFEM Version Information</h1>
<p>version={html.escape(str(product.get('version', '')))}, python={html.escape(str(product.get('python', '')))}, platform={html.escape(str(product.get('platform', '')))}</p>
<p>passed={html.escape(str(info.get('passed', '')))}, errors={html.escape(str(info.get('error_count', '')))}, warnings={html.escape(str(info.get('warning_count', '')))}</p>
<h2>Distribution Notice</h2>
<ul>
<li>{html.escape(str(notice.get('dwg_policy', '')))}</li>
<li>{html.escape(str(notice.get('font_policy', '')))}</li>
<li>{html.escape(str(notice.get('license_policy', '')))}</li>
<li>{html.escape(str(notice.get('update_policy', '')))}</li>
</ul>
<h2>Dependencies And External Tools</h2>
<table><thead><tr><th>category</th><th>name</th><th>status</th><th>required</th><th>version</th><th>license</th><th>purpose</th><th>path</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Fonts</h2>
<p>status={html.escape(str(fonts.get('status', '')))}, available={html.escape(', '.join(str(item) for item in fonts.get('available_preferred', [])))}</p>
</body></html>
"""


def _license_from_metadata(meta: Mapping[str, Any]) -> tuple[str, str]:
    license_text = str(meta.get("License", "") or "").strip()
    if license_text and license_text.upper() != "UNKNOWN":
        return license_text, "License"
    classifiers = [str(item) for item in meta.get_all("Classifier", []) if "License" in str(item)] if hasattr(meta, "get_all") else []
    if classifiers:
        return "; ".join(classifiers), "Classifier"
    expression = str(meta.get("License-Expression", "") or "").strip()
    if expression:
        return expression, "License-Expression"
    return "", ""


def _metadata(package: str) -> Mapping[str, Any]:
    try:
        return metadata.metadata(package)
    except Exception:
        return {}


def _metadata_version(package: str, module_name: str) -> str:
    try:
        return metadata.version(package)
    except Exception:
        try:
            module = importlib.import_module(module_name)
            return str(getattr(module, "__version__", "unknown"))
        except Exception:
            return ""


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _module_path(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__file__", "") or "")
    except Exception:
        return ""


__all__ = [
    "PREFERRED_GUI_FONTS",
    "PYTHON_DEPENDENCIES",
    "build_version_info",
    "write_version_info_artifacts",
]
