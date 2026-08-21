"""Version-update compatibility, input migration, and artifact revalidation."""

from __future__ import annotations

import copy
import csv
import html
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from . import __version__
from .input_diagnostics import diagnose_input_config
from .public_artifacts import write_html_artifact, write_json_artifact


CONFIG_SCHEMA_VERSION = "geofem.input_config.v1"
UPDATE_COMPATIBILITY_SCHEMA = "geofem.update_compatibility.v1"
REQUIRED_ARTIFACTS = ("summary.json", "case_manifest.json")
RECOMMENDED_ARTIFACTS = ("result_view_index.json", "standard_report.html", "calculation_report_manifest.json", "reliability_summary.json")


def build_update_compatibility_report(
    input_config: Mapping[str, Any],
    *,
    previous_version: str | None = None,
    artifact_dir: str | Path | None = None,
    current_version: str = __version__,
) -> dict[str, Any]:
    """Build a local update/migration report without requiring an online updater."""

    migration = migrate_input_config(input_config)
    migrated_config = migration["migrated_config"]
    diagnostics = diagnose_input_config(migrated_config)
    artifact_report = revalidate_generated_artifacts(artifact_dir, migration_changed=bool(migration["changed"]))
    version = _version_delta(previous_version, current_version)
    migration_errors = 1 if int(diagnostics.get("error_count", 0) or 0) > 0 else 0
    artifact_errors = int(artifact_report.get("error_count", 0) or 0)
    warning_count = (
        int(diagnostics.get("warning_count", 0) or 0)
        + int(artifact_report.get("warning_count", 0) or 0)
        + (1 if version["status"] in {"newer_current", "older_current"} else 0)
    )
    return {
        "schema": UPDATE_COMPATIBILITY_SCHEMA,
        "features": [
            "version_delta_check",
            "input_config_schema_migration",
            "diagnostics_after_migration",
            "generated_artifact_revalidation",
            "migration_guide_artifacts",
        ],
        "passed": migration_errors == 0 and artifact_errors == 0,
        "error_count": migration_errors + artifact_errors,
        "warning_count": warning_count,
        "version": version,
        "config_schema": {
            "source": migration["source_schema"],
            "target": CONFIG_SCHEMA_VERSION,
            "status": "migrated" if migration["changed"] else "current_or_compatible",
        },
        "migration": migration,
        "diagnostics": diagnostics,
        "artifact_revalidation": artifact_report,
        "recommended_actions": _recommended_actions(migration, diagnostics, artifact_report, version),
    }


def migrate_input_config(input_config: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate legacy-but-public GeoFEM input spellings to the current schema."""

    cfg = copy.deepcopy(dict(input_config))
    source_schema = str(cfg.get("schema") or cfg.get("schema_version") or cfg.get("format_version") or "unknown")
    actions: list[dict[str, Any]] = []

    _ensure_analysis_defaults(cfg, actions)
    _move_top_level(cfg, "boundary", "boundary_conditions", actions, coerce_list=True)
    _move_top_level(cfg, "boundaries", "boundary_conditions", actions, coerce_list=True)
    _move_top_level(cfg, "constraints", "boundary_conditions", actions, coerce_list=True)
    _move_top_level(cfg, "load", "loads", actions, coerce_list=True)
    _move_top_level(cfg, "stage", "stages", actions, coerce_list=True)
    _move_top_level(cfg, "solver_settings", "solver", actions, coerce_list=False)
    _move_top_level(cfg, "postprocess", "post", actions, coerce_list=False)
    _migrate_material(cfg, actions)
    _migrate_mesh(cfg, actions)
    _migrate_output_dir(cfg, actions)

    cfg["schema"] = CONFIG_SCHEMA_VERSION
    cfg.pop("schema_version", None)
    cfg.pop("format_version", None)
    if source_schema != CONFIG_SCHEMA_VERSION:
        actions.append(
            {
                "id": "schema.target",
                "severity": "INFO",
                "path": "schema",
                "action": f"set schema to {CONFIG_SCHEMA_VERSION}",
            }
        )
    return {
        "source_schema": source_schema,
        "target_schema": CONFIG_SCHEMA_VERSION,
        "changed": bool(actions),
        "action_count": len(actions),
        "actions": actions,
        "migrated_config": cfg,
    }


def revalidate_generated_artifacts(artifact_dir: str | Path | None, *, migration_changed: bool = False) -> dict[str, Any]:
    """Check whether existing result artifacts should be trusted after an update."""

    if artifact_dir is None:
        return {
            "status": "not_requested",
            "passed": True,
            "error_count": 0,
            "warning_count": 0,
            "artifact_dir": "",
            "checks": [],
        }
    root = Path(artifact_dir)
    checks: list[dict[str, Any]] = []
    if not root.exists():
        return {
            "status": "missing_artifact_dir",
            "passed": False,
            "error_count": 1,
            "warning_count": 0,
            "artifact_dir": str(root),
            "checks": [{"artifact": str(root), "status": "ERROR", "detail": "artifact directory does not exist"}],
        }
    for name in REQUIRED_ARTIFACTS:
        _artifact_check(root, name, "ERROR", checks)
    for name in RECOMMENDED_ARTIFACTS:
        _artifact_check(root, name, "WARN", checks)
    case_manifest = _read_json(root / "case_manifest.json")
    if case_manifest and str(case_manifest.get("status", "")) != "completed":
        checks.append({"artifact": "case_manifest.json", "status": "WARN", "detail": f"case status is {case_manifest.get('status')!r}; rerun is recommended"})
    if migration_changed:
        checks.append({"artifact": str(root), "status": "WARN", "detail": "input schema migration changed the model; rerun analysis to refresh generated artifacts"})
    errors = sum(1 for row in checks if row["status"] == "ERROR")
    warnings = sum(1 for row in checks if row["status"] == "WARN")
    return {
        "status": "passed" if errors == 0 else "failed",
        "passed": errors == 0,
        "error_count": errors,
        "warning_count": warnings,
        "artifact_dir": str(root),
        "checks": checks,
    }


def write_update_compatibility_artifacts(
    input_config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    previous_version: str | None = None,
    artifact_dir: str | Path | None = None,
    current_version: str = __version__,
) -> dict[str, str]:
    """Write update compatibility JSON/CSV/HTML, migration guide, and migrated YAML."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_update_compatibility_report(
        input_config,
        previous_version=previous_version,
        artifact_dir=artifact_dir,
        current_version=current_version,
    )
    paths = {
        "json": str(out / "update_compatibility.json"),
        "csv": str(out / "update_compatibility.csv"),
        "html": str(out / "update_compatibility.html"),
        "guide": str(out / "update_migration_guide.md"),
        "migrated_input": str(out / "migrated_input.yaml"),
    }
    write_json_artifact(paths["json"], report)
    _write_rows_csv(report, Path(paths["csv"]))
    write_html_artifact(paths["html"], _compatibility_html(report))
    Path(paths["guide"]).write_text(_migration_guide_markdown(report), encoding="utf-8")
    Path(paths["migrated_input"]).write_text(yaml.safe_dump(report["migration"]["migrated_config"], allow_unicode=True, sort_keys=False), encoding="utf-8")
    return paths


def _ensure_analysis_defaults(cfg: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    analysis = cfg.get("analysis")
    if not isinstance(analysis, dict):
        cfg["analysis"] = {"dimension": "2D", "type": "static_plane_strain"}
        actions.append({"id": "analysis.create", "severity": "INFO", "path": "analysis", "action": "created default 2D plane-strain analysis block"})
        return
    if "dimension" not in analysis:
        analysis["dimension"] = "2D"
        actions.append({"id": "analysis.dimension", "severity": "INFO", "path": "analysis.dimension", "action": "defaulted to 2D"})
    if "type" not in analysis:
        analysis["type"] = "static_plane_strain"
        actions.append({"id": "analysis.type", "severity": "INFO", "path": "analysis.type", "action": "defaulted to static_plane_strain"})
    if str(analysis.get("type", "")).lower() == "plane_strain":
        analysis["type"] = "static_plane_strain"
        actions.append({"id": "analysis.type.rename", "severity": "INFO", "path": "analysis.type", "action": "renamed plane_strain to static_plane_strain"})


def _move_top_level(cfg: dict[str, Any], old: str, new: str, actions: list[dict[str, Any]], *, coerce_list: bool) -> None:
    if old not in cfg or new in cfg:
        return
    value = cfg.pop(old)
    cfg[new] = _as_list(value) if coerce_list else value
    actions.append({"id": f"{old}.rename", "severity": "INFO", "path": new, "action": f"renamed {old} to {new}"})


def _migrate_material(cfg: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    if "materials" in cfg or "material" not in cfg:
        return
    raw = cfg.pop("material")
    if isinstance(raw, Mapping) and _looks_like_single_material(raw):
        cfg["materials"] = {"soil": dict(raw)}
        action = "converted single material block to materials.soil"
    elif isinstance(raw, Mapping):
        cfg["materials"] = dict(raw)
        action = "renamed material mapping to materials"
    else:
        cfg["materials"] = {"soil": {"model": "elastic"}}
        action = "created placeholder materials.soil because legacy material was not a mapping"
    actions.append({"id": "material.rename", "severity": "INFO", "path": "materials", "action": action})


def _migrate_mesh(cfg: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    mesh = cfg.get("mesh")
    if not isinstance(mesh, dict):
        return
    if "element_type" not in mesh:
        for old in ("type", "element"):
            if old in mesh:
                mesh["element_type"] = mesh.pop(old)
                actions.append({"id": f"mesh.{old}.rename", "severity": "INFO", "path": "mesh.element_type", "action": f"renamed mesh.{old} to mesh.element_type"})
                break
    if "material" not in mesh and "materials" in cfg:
        materials = cfg.get("materials")
        if isinstance(materials, Mapping) and len(materials) == 1:
            mesh["material"] = next(iter(materials.keys()))
            actions.append({"id": "mesh.material.default", "severity": "INFO", "path": "mesh.material", "action": "selected the only material as mesh.material"})


def _migrate_output_dir(cfg: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    if "output_dir" not in cfg:
        return
    output = cfg.get("output")
    if not isinstance(output, dict):
        output = {}
        cfg["output"] = output
    if "directory" not in output:
        output["directory"] = cfg.pop("output_dir")
        actions.append({"id": "output.directory", "severity": "INFO", "path": "output.directory", "action": "moved output_dir to output.directory"})


def _artifact_check(root: Path, name: str, missing_severity: str, checks: list[dict[str, Any]]) -> None:
    path = root / name
    if not path.exists():
        checks.append({"artifact": name, "status": missing_severity, "detail": "missing generated artifact"})
        return
    if path.suffix.lower() == ".json" and _read_json(path) is None:
        checks.append({"artifact": name, "status": "ERROR", "detail": "JSON artifact exists but is not readable"})
        return
    checks.append({"artifact": name, "status": "OK", "detail": f"{path.stat().st_size} bytes"})


def _version_delta(previous_version: str | None, current_version: str) -> dict[str, Any]:
    previous = str(previous_version or "")
    current = str(current_version or __version__)
    if not previous:
        status = "current_unknown_previous"
    elif _version_tuple(previous) == _version_tuple(current):
        status = "same_version"
    elif _version_tuple(previous) < _version_tuple(current):
        status = "newer_current"
    else:
        status = "older_current"
    return {
        "previous": previous,
        "current": current,
        "status": status,
        "online_update_check": "not_required_for_local_distribution",
        "config_schema": CONFIG_SCHEMA_VERSION,
    }


def _recommended_actions(
    migration: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifact_report: Mapping[str, Any],
    version: Mapping[str, Any],
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if version.get("status") in {"newer_current", "older_current"}:
        actions.append({"priority": "medium", "action": "review update_migration_guide.md before distributing the updated project"})
    if migration.get("changed"):
        actions.append({"priority": "high", "action": "review migrated_input.yaml and rerun diagnostics/analysis"})
    if int(diagnostics.get("error_count", 0) or 0) > 0:
        actions.append({"priority": "high", "action": "fix migrated input diagnostics before solving"})
    if artifact_report.get("status") not in {"passed", "not_requested"}:
        actions.append({"priority": "high", "action": "regenerate missing or unreadable result artifacts"})
    elif int(artifact_report.get("warning_count", 0) or 0) > 0:
        actions.append({"priority": "medium", "action": "rerun analysis to refresh recommended Post/report artifacts"})
    if not actions:
        actions.append({"priority": "info", "action": "no migration or artifact refresh is required"})
    return actions


def _write_rows_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = ["section", "id", "status", "severity", "path", "artifact", "action", "detail"]
    rows: list[dict[str, Any]] = []
    for row in report.get("migration", {}).get("actions", []):
        if isinstance(row, Mapping):
            rows.append({"section": "migration", **{field: row.get(field, "") for field in fields if field != "section"}})
    for row in report.get("artifact_revalidation", {}).get("checks", []):
        if isinstance(row, Mapping):
            rows.append({"section": "artifact", "status": row.get("status", ""), "artifact": row.get("artifact", ""), "detail": row.get("detail", "")})
    for row in report.get("recommended_actions", []):
        if isinstance(row, Mapping):
            rows.append({"section": "recommended_action", "severity": row.get("priority", ""), "action": row.get("action", "")})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _compatibility_html(report: Mapping[str, Any]) -> str:
    actions = _html_rows(report.get("migration", {}).get("actions", []), ("id", "severity", "path", "action"))
    checks = _html_rows(report.get("artifact_revalidation", {}).get("checks", []), ("artifact", "status", "detail"))
    recommended = _html_rows(report.get("recommended_actions", []), ("priority", "action"))
    version = report.get("version", {}) if isinstance(report.get("version", {}), Mapping) else {}
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM Update Compatibility</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ccd2d8;padding:6px;text-align:left;vertical-align:top}}th{{background:#eef2f7}}</style></head>
<body>
<h1>GeoFEM Update Compatibility</h1>
<p>passed={html.escape(str(report.get('passed')))}, errors={html.escape(str(report.get('error_count')))}, warnings={html.escape(str(report.get('warning_count')))}</p>
<p>version: {html.escape(str(version.get('previous', '')))} -> {html.escape(str(version.get('current', '')))} / status={html.escape(str(version.get('status', '')))}</p>
<h2>Migration Actions</h2>{actions}
<h2>Artifact Revalidation</h2>{checks}
<h2>Recommended Actions</h2>{recommended}
</body></html>
"""


def _migration_guide_markdown(report: Mapping[str, Any]) -> str:
    version = report.get("version", {}) if isinstance(report.get("version", {}), Mapping) else {}
    lines = [
        "# GeoFEM 更新・移行ガイド",
        "",
        f"- 判定: `{report.get('passed')}`",
        f"- エラー: `{report.get('error_count')}`",
        f"- 警告: `{report.get('warning_count')}`",
        f"- バージョン: `{version.get('previous', '')}` -> `{version.get('current', '')}`",
        f"- 設定スキーマ: `{report.get('config_schema', {}).get('source', '')}` -> `{report.get('config_schema', {}).get('target', '')}`",
        "",
        "## 推奨作業",
        "",
    ]
    for row in report.get("recommended_actions", []):
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('priority', '')}` {row.get('action', '')}")
    lines.extend(["", "## 移行内容", ""])
    for row in report.get("migration", {}).get("actions", []):
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('path', '')}` {row.get('action', '')}")
    lines.extend(["", "## 成果物再検証", ""])
    for row in report.get("artifact_revalidation", {}).get("checks", []):
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('status', '')}` {row.get('artifact', '')}: {row.get('detail', '')}")
    return "\n".join(lines) + "\n"


def _html_rows(rows: Any, fields: tuple[str, ...]) -> str:
    body = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>")
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _looks_like_single_material(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("model", "E", "nu", "gamma", "cohesion", "friction_angle", "permeability"))


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(match) for match in re.findall(r"\d+", value)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RECOMMENDED_ARTIFACTS",
    "REQUIRED_ARTIFACTS",
    "UPDATE_COMPATIBILITY_SCHEMA",
    "build_update_compatibility_report",
    "migrate_input_config",
    "revalidate_generated_artifacts",
    "write_update_compatibility_artifacts",
]
