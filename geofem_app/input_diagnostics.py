"""Input diagnostics shared by the CLI and GUI layers."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fem2d_types import SUPPORTED_2D_CORE_STAGE_TYPES, SUPPORTED_ELEMENTS, SUPPORTED_INTEGRATION
from .input_assistance import build_input_assistance_summary, write_input_assistance_artifacts
from .material_models import material_validation_issues
from .messages import DEFAULT_LOCALE, message


TOP_LEVEL_KEYS = {
    "schema",
    "metadata",
    "analysis",
    "model",
    "geometry",
    "mesh",
    "materials",
    "material",
    "sets",
    "node_sets",
    "element_sets",
    "boundary_conditions",
    "bc",
    "loads",
    "steps",
    "stages",
    "solver",
    "output",
    "post",
    "report",
    "calculation_report",
    "organization_profile",
    "customization",
    "template_variants",
    "interfaces",
    "structural_elements",
    "mpc_constraints",
    "mpc",
    "load_combinations",
    "vgflow",
    "vgflow2d",
    "checks",
}


def diagnose_input_config(cfg: Mapping[str, Any], *, locale: str = DEFAULT_LOCALE) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def msg(message_key: str, **values: Any) -> str:
        return message(message_key, locale=locale, **values)

    def add(severity: str, path: str, text_key: str, suggestion_key: str = "", context: Mapping[str, Any] | None = None, **values: Any) -> None:
        issues.append(
            {
                "severity": severity,
                "path": path,
                "message": msg(text_key, **values) if text_key else "",
                "suggestion": msg(suggestion_key, **values) if suggestion_key else "",
                "context": dict(context or {}),
            }
        )

    def add_issue(issue: Mapping[str, Any]) -> None:
        issues.append(
            {
                "severity": str(issue.get("severity", "ERROR")),
                "path": str(issue.get("path", "")),
                "message": str(issue.get("message", "")),
                "suggestion": str(issue.get("suggestion", "")),
                "context": dict(issue.get("context", {}) if isinstance(issue.get("context", {}), Mapping) else {}),
            }
        )

    for key in sorted(str(key) for key in cfg):
        if key not in TOP_LEVEL_KEYS:
            add(
                "WARN",
                key,
                "diagnostics.unused_top_level_key.message",
                "diagnostics.unused_top_level_key.suggestion",
            )

    analysis = _mapping(cfg.get("analysis", {}))
    dimension = str(analysis.get("dimension", "2D")).upper().replace(" ", "")
    if dimension != "2D":
        add("ERROR", "analysis.dimension", "diagnostics.dimension.message", "diagnostics.dimension.suggestion", {"value": dimension})
    unit_system = str(analysis.get("unit_system", analysis.get("units", "m-kN")))
    if unit_system.lower() not in {"m-kn", "m-n", "si", "mm-n", "user", "custom"}:
        add("WARN", "analysis.unit_system", "diagnostics.unit_system.message", "diagnostics.unit_system.suggestion", {"value": unit_system})

    mesh = _mapping(cfg.get("mesh", {}))
    if not mesh:
        add("ERROR", "mesh", "diagnostics.mesh.missing.message", "diagnostics.mesh.missing.suggestion")
    _diagnose_mesh(mesh, add)

    materials = _mapping(cfg.get("materials", cfg.get("material", {})))
    if not materials:
        add("ERROR", "materials", "diagnostics.materials.missing.message", "diagnostics.materials.missing.suggestion")
    _diagnose_materials(materials, add, add_issue)

    material_names = {str(name) for name in materials}
    _diagnose_material_references(mesh, material_names, add)

    node_sets, element_sets = _declared_sets(cfg, mesh)
    _diagnose_boundary_conditions(_ensure_list(cfg.get("boundary_conditions", cfg.get("bc", []))), node_sets, add, "boundary_conditions")
    _diagnose_loads(_ensure_list(cfg.get("loads", [])), node_sets, element_sets, add, "loads")
    _diagnose_stages(_ensure_list(cfg.get("stages", cfg.get("steps", []))), node_sets, element_sets, add)
    _diagnose_constraints(_ensure_list(cfg.get("boundary_conditions", cfg.get("bc", []))), node_sets, mesh, add)
    input_assistance = build_input_assistance_summary(cfg, locale=locale)
    for row in input_assistance.get("diagnostics", []):
        if not isinstance(row, Mapping):
            continue
        severity = str(row.get("severity", "INFO"))
        if severity not in {"ERROR", "WARN"}:
            continue
        add_issue(
            {
                "severity": severity,
                "path": row.get("path", ""),
                "message": row.get("message", ""),
                "suggestion": row.get("suggestion", ""),
                "context": {"source": "input_assistance"},
            }
        )

    errors = sum(1 for issue in issues if issue["severity"] == "ERROR")
    warnings = sum(1 for issue in issues if issue["severity"] == "WARN")
    return {
        "schema": "geofem.input_diagnostics.v1",
        "locale": locale,
        "passed": errors == 0,
        "error_count": errors,
        "warning_count": warnings,
        "issue_count": len(issues),
        "input_assistance": input_assistance,
        "issues": issues,
    }


def write_input_diagnostics(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "input_diagnostics.json"
    csv_path = out / "input_diagnostics.csv"
    html_path = out / "input_diagnostics.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["severity", "path", "message", "suggestion"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for issue in summary.get("issues", []):
            if isinstance(issue, Mapping):
                writer.writerow({field: issue.get(field, "") for field in fields})
    html_path.write_text(_diagnostics_html(summary), encoding="utf-8")
    artifacts = {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}
    assistance = summary.get("input_assistance", {})
    if isinstance(assistance, Mapping):
        assistance_paths = write_input_assistance_artifacts(assistance, out)
        artifacts.update({f"input_assistance_{key}": value for key, value in assistance_paths.items()})
    return artifacts


def _diagnose_mesh(mesh: Mapping[str, Any], add: Any) -> None:
    generator = str(mesh.get("generator", "")).lower().strip()
    nodes = mesh.get("nodes")
    elements = mesh.get("elements")
    if not generator and not (isinstance(nodes, Mapping) and isinstance(elements, Sequence)):
        add("ERROR", "mesh", "diagnostics.mesh.generator_or_explicit.message", "diagnostics.mesh.generator_or_explicit.suggestion")
    etype = str(mesh.get("element_type", mesh.get("type", ""))).upper()
    if etype and etype not in SUPPORTED_ELEMENTS:
        add(
            "ERROR",
            "mesh.element_type",
            "diagnostics.mesh.unsupported_element.message",
            "diagnostics.mesh.unsupported_element.suggestion",
            {"value": etype},
            supported=", ".join(sorted(SUPPORTED_ELEMENTS)),
        )
    integration = str(mesh.get("integration", "")).upper()
    if integration and integration not in SUPPORTED_INTEGRATION:
        add(
            "WARN",
            "mesh.integration",
            "diagnostics.mesh.unsupported_integration.message",
            "diagnostics.mesh.unsupported_integration.suggestion",
            {"value": integration},
            supported=", ".join(sorted(SUPPORTED_INTEGRATION)),
        )
    for name in ("nx", "ny"):
        if name in mesh:
            try:
                if int(mesh[name]) <= 0:
                    add("ERROR", f"mesh.{name}", "diagnostics.mesh.division_positive.message", "diagnostics.mesh.division_positive.suggestion", name=name)
            except (TypeError, ValueError):
                add("ERROR", f"mesh.{name}", "diagnostics.mesh.division_integer.message", "diagnostics.mesh.division_integer.suggestion", name=name)


def _diagnose_materials(materials: Mapping[str, Any], add: Any, add_issue: Any) -> None:
    for name, raw in materials.items():
        path = f"materials.{name}"
        mat = _mapping(raw)
        if not mat:
            add("ERROR", path, "diagnostics.material.mapping.message", "diagnostics.material.mapping.suggestion")
            continue
        for issue in material_validation_issues(str(name), mat):
            add_issue(issue)
        for key in ("E", "nu"):
            if key not in mat:
                add("ERROR", f"{path}.{key}", "diagnostics.material.required.message", "diagnostics.material.required.suggestion", key=key)
        for key in ("E", "thickness"):
            if key in mat and _float_or_none(mat[key]) is not None and float(mat[key]) <= 0.0:
                add("ERROR", f"{path}.{key}", "diagnostics.material.positive.message", "diagnostics.material.positive.suggestion", key=key)
        if "nu" in mat:
            value = _float_or_none(mat["nu"])
            if value is not None and not 0.0 <= value < 0.5:
                add("ERROR", f"{path}.nu", "diagnostics.material.nu_range.message", "diagnostics.material.nu_range.suggestion")


def _diagnose_material_references(mesh: Mapping[str, Any], material_names: set[str], add: Any) -> None:
    default_material = mesh.get("material")
    if default_material is not None and str(default_material) not in material_names:
        add("ERROR", "mesh.material", "diagnostics.mesh.material_missing.message", "diagnostics.mesh.material_missing.suggestion", {"material": default_material})
    elements = mesh.get("elements")
    if isinstance(elements, Sequence) and not isinstance(elements, (str, bytes)):
        for index, raw in enumerate(elements):
            element = _mapping(raw)
            material = element.get("material", default_material)
            if material is not None and str(material) not in material_names:
                add(
                    "ERROR",
                    f"mesh.elements[{index}].material",
                    "diagnostics.mesh.element_material_missing.message",
                    "diagnostics.mesh.element_material_missing.suggestion",
                    {"material": material},
                )


def _diagnose_boundary_conditions(items: list[Any], node_sets: set[str], add: Any, prefix: str) -> None:
    for index, raw in enumerate(items):
        item = _mapping(raw)
        if not item:
            add("ERROR", f"{prefix}[{index}]", "diagnostics.bc.mapping.message", "diagnostics.bc.mapping.suggestion")
            continue
        target = item.get("set")
        if target is not None and str(target) not in node_sets:
            add("ERROR", f"{prefix}[{index}].set", "diagnostics.bc.unknown_set.message", "diagnostics.bc.unknown_set.suggestion", {"set": target})
        if not any(key in item for key in ("fixed", "ux", "uy", "x", "y")):
            add("WARN", f"{prefix}[{index}]", "diagnostics.bc.no_component.message", "diagnostics.bc.no_component.suggestion")


def _diagnose_loads(items: list[Any], node_sets: set[str], element_sets: set[str], add: Any, prefix: str) -> None:
    for index, raw in enumerate(items):
        item = _mapping(raw)
        if not item:
            add("ERROR", f"{prefix}[{index}]", "diagnostics.loads.mapping.message", "diagnostics.loads.mapping.suggestion")
            continue
        load_type = str(item.get("type", "")).lower().strip()
        is_body_load = load_type in {"gravity", "self_weight", "body"} or bool(item.get("self_weight", False))
        body_keys = ("gx", "gy", "bx", "by", "body_fx", "body_fy", "body_x", "body_y")
        target = item.get("set")
        if target is not None and str(target) not in node_sets and str(target) not in element_sets:
            add("ERROR", f"{prefix}[{index}].set", "diagnostics.loads.unknown_set.message", "diagnostics.loads.unknown_set.suggestion", {"set": target})
        if not is_body_load and not any(key in item for key in ("fx", "fy", "tx", "ty", "px", "py", *body_keys, "edge", "node", "nodes", "set")):
            add("WARN", f"{prefix}[{index}]", "diagnostics.loads.no_component.message", "diagnostics.loads.no_component.suggestion")


def _diagnose_stages(stages: list[Any], node_sets: set[str], element_sets: set[str], add: Any) -> None:
    for index, raw in enumerate(stages):
        stage = _mapping(raw)
        if not stage:
            add("ERROR", f"stages[{index}]", "diagnostics.stage.mapping.message", "diagnostics.stage.mapping.suggestion")
            continue
        stage_type = str(stage.get("type", "")).lower().strip()
        if stage_type not in SUPPORTED_2D_CORE_STAGE_TYPES:
            add("ERROR", f"stages[{index}].type", "diagnostics.stage.unsupported_type.message", "diagnostics.stage.unsupported_type.suggestion", {"type": stage_type})
        _diagnose_boundary_conditions(_ensure_list(stage.get("boundary_conditions", stage.get("bc", []))), node_sets, add, f"stages[{index}].boundary_conditions")
        _diagnose_loads(_ensure_list(stage.get("loads", [])), node_sets, element_sets, add, f"stages[{index}].loads")
        target_set = stage.get("set")
        if target_set is not None and str(target_set) not in element_sets and str(target_set) not in node_sets:
            add("ERROR", f"stages[{index}].set", "diagnostics.stage.unknown_set.message", "diagnostics.stage.unknown_set.suggestion", {"set": target_set})


def _diagnose_constraints(items: list[Any], node_sets: set[str], mesh: Mapping[str, Any], add: Any) -> None:
    _ = node_sets
    if not items:
        add("WARN", "boundary_conditions", "diagnostics.constraints.none.message", "diagnostics.constraints.none.suggestion")
        return
    constrained_x = False
    constrained_y = False
    fixed_all = False
    for raw in items:
        item = _mapping(raw)
        fixed_all = fixed_all or bool(item.get("fixed", False))
        constrained_x = constrained_x or "ux" in item or "x" in item or fixed_all
        constrained_y = constrained_y or "uy" in item or "y" in item or fixed_all
    if not constrained_x:
        add("WARN", "boundary_conditions", "diagnostics.constraints.no_x.message", "diagnostics.constraints.no_x.suggestion")
    if not constrained_y:
        add("WARN", "boundary_conditions", "diagnostics.constraints.no_y.message", "diagnostics.constraints.no_y.suggestion")
    if fixed_all and str(mesh.get("generator", "")).lower() == "rectangle" and not _ensure_list(mesh.get("elements", [])):
        add("WARN", "boundary_conditions", "diagnostics.constraints.over_fixed.message", "diagnostics.constraints.over_fixed.suggestion")


def _declared_sets(cfg: Mapping[str, Any], mesh: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    node_sets = {"all", "left", "right", "top", "bottom"}
    element_sets = {"all"}
    sets = _mapping(cfg.get("sets", {}))
    node_sets.update(str(key) for key in _mapping(sets.get("nodes", {})))
    element_sets.update(str(key) for key in _mapping(sets.get("elements", {})))
    node_sets.update(str(key) for key in _mapping(cfg.get("node_sets", {})))
    element_sets.update(str(key) for key in _mapping(cfg.get("element_sets", {})))
    mesh_sets = _mapping(mesh.get("sets", {}))
    node_sets.update(str(key) for key in _mapping(mesh_sets.get("nodes", {})))
    element_sets.update(str(key) for key in _mapping(mesh_sets.get("elements", {})))
    return node_sets, element_sets


def _diagnostics_html(summary: Mapping[str, Any]) -> str:
    locale = str(summary.get("locale", DEFAULT_LOCALE))
    rows = []
    for issue in summary.get("issues", []):
        if not isinstance(issue, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(issue.get('severity', '')))}</td>"
            f"<td>{html.escape(str(issue.get('path', '')))}</td>"
            f"<td>{html.escape(str(issue.get('message', '')))}</td>"
            f"<td>{html.escape(str(issue.get('suggestion', '')))}</td>"
            "</tr>"
        )
    title = message("diagnostics.html.title", locale=locale)
    heading = message("diagnostics.html.heading", locale=locale)
    return f"""<!doctype html>
<html lang="{html.escape(locale)}"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>{html.escape(heading)}</h1>
<p>passed={bool(summary.get('passed', False))}, errors={int(summary.get('error_count', 0) or 0)}, warnings={int(summary.get('warning_count', 0) or 0)}</p>
<table><thead><tr><th>severity</th><th>path</th><th>message</th><th>suggestion</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["TOP_LEVEL_KEYS", "diagnose_input_config", "write_input_diagnostics"]
