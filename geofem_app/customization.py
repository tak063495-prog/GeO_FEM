"""Organization-level customization profiles for GeoFEM projects."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


ORGANIZATION_PROFILE_SCHEMA = "geofem.organization_profile.v1"
PROJECT_TEMPLATE_CATALOG_SCHEMA = "geofem.project_template_catalog.v1"
CUSTOMIZATION_VALIDATION_SCHEMA = "geofem.customization_validation.v1"
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def default_organization_profile(organization_name: str = "GeoFEM Organization") -> dict[str, Any]:
    """Return an editable organization profile with practical defaults."""

    return {
        "schema": ORGANIZATION_PROFILE_SCHEMA,
        "profile_id": "default_organization",
        "revision": "2026.05",
        "organization": {
            "name": organization_name,
            "department": "Geotechnical Engineering",
            "prepared_by": "",
            "checked_by": "",
            "approved_by": "",
        },
        "branding": {
            "logo_path": "templates/logo.png",
            "logo_alt": organization_name,
            "colors": {
                "primary": "#1f5f8b",
                "secondary": "#2f7d32",
                "accent": "#c57b00",
                "surface": "#f7fafc",
                "error": "#b3261e",
                "warning": "#b26a00",
                "ok": "#1b7f3a",
            },
        },
        "units": {
            "unit_system": "m-kN",
            "length": "m",
            "force": "kN",
            "stress": "kPa",
            "density": "kN/m3",
        },
        "defaults": {
            "project_template": "geofem_review",
            "report_template": "organization_review_a4",
            "post_palette": "organization_safety",
            "element_type": "QUAD4",
            "integration": "B-bar",
        },
        "report_templates": [
            {
                "id": "organization_review_a4",
                "template_id": "organization_review_a4",
                "template_name": "Organization review A4",
                "template_revision": "2026.05",
                "title": "GeoFEM 2D 計算書",
                "subtitle": "解析条件・結果・照査記録",
                "page_style": "A4 portrait titleblock",
            },
            {
                "id": "client_submission_a3",
                "template_id": "client_submission_a3",
                "template_name": "Client submission A3 landscape",
                "template_revision": "2026.05",
                "title": "GeoFEM 2D 提出図書",
                "subtitle": "施工段階解析・Post図・判定表",
                "page_style": "A3 landscape titleblock",
            },
        ],
        "project_templates": _default_project_templates(),
    }


def load_organization_profile(path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML organization profile."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError("organization profile root must be a mapping")
    return dict(data)


def validate_organization_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate required organization customization fields."""

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if profile.get("schema") not in {"", None, ORGANIZATION_PROFILE_SCHEMA}:
        errors.append({"path": "schema", "message": f"unsupported schema: {profile.get('schema')}"})
    organization = _mapping(profile.get("organization"))
    if not organization.get("name"):
        errors.append({"path": "organization.name", "message": "organization name is required"})
    units = _mapping(profile.get("units"))
    if not units.get("unit_system"):
        errors.append({"path": "units.unit_system", "message": "unit system is required"})
    colors = _mapping(_mapping(profile.get("branding")).get("colors"))
    for key in ("primary", "secondary", "accent", "error", "warning", "ok"):
        value = str(colors.get(key, ""))
        if not HEX_COLOR_RE.match(value):
            errors.append({"path": f"branding.colors.{key}", "message": "color must be #RRGGBB"})
    template_ids = [str(row.get("id", "")) for row in _list_of_mappings(profile.get("project_templates"))]
    if len(template_ids) != len(set(template_ids)):
        errors.append({"path": "project_templates", "message": "project template id is duplicated"})
    report_ids = [str(row.get("id", row.get("template_id", ""))) for row in _list_of_mappings(profile.get("report_templates"))]
    if len(report_ids) != len(set(report_ids)):
        errors.append({"path": "report_templates", "message": "report template id is duplicated"})
    default_template = str(_mapping(profile.get("defaults")).get("project_template", ""))
    if default_template and default_template not in set(template_ids):
        warnings.append({"path": "defaults.project_template", "message": f"default template is not in catalog: {default_template}"})
    return {
        "schema": CUSTOMIZATION_VALIDATION_SCHEMA,
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "feature_count": len(project_template_catalog(profile)),
    }


def project_template_catalog(profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return project templates separated from organization branding data."""

    source = profile or default_organization_profile()
    templates = _list_of_mappings(source.get("project_templates"))
    if not templates:
        templates = _default_project_templates()
    rows: list[dict[str, Any]] = []
    for row in templates:
        rows.append(
            {
                "id": str(row.get("id", "")),
                "label": str(row.get("label", row.get("id", ""))),
                "domain": str(row.get("domain", "")),
                "solver_route": str(row.get("solver_route", "GeoFEM 2D")),
                "unit_system": str(_mapping(row.get("analysis_defaults")).get("unit_system", _mapping(source.get("units")).get("unit_system", ""))),
                "report_template": str(_mapping(_mapping(row.get("report_defaults")).get("template")).get("id", "")),
                "description": str(row.get("description", "")),
            }
        )
    return rows


def apply_organization_profile(
    cfg: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    *,
    template_id: str | None = None,
) -> dict[str, Any]:
    """Apply organization defaults without overwriting explicit project settings."""

    organization_profile = profile or default_organization_profile()
    out = copy.deepcopy(dict(cfg))
    defaults = _mapping(organization_profile.get("defaults"))
    selected_template = template_id or str(defaults.get("project_template", ""))
    template = _template_by_id(organization_profile, selected_template)
    if template:
        _apply_template_defaults(out, template)
    analysis = out.setdefault("analysis", {})
    if not isinstance(analysis, dict):
        analysis = {}
        out["analysis"] = analysis
    units = _mapping(organization_profile.get("units"))
    _merge_defaults(analysis, {"unit_system": units.get("unit_system", "m-kN"), "dimension": "2D"})
    branding = _mapping(organization_profile.get("branding"))
    organization = _mapping(organization_profile.get("organization"))
    colors = _mapping(branding.get("colors"))
    report = out.setdefault("report", {})
    if not isinstance(report, dict):
        report = {}
        out["report"] = report
    _merge_defaults(
        report,
        {
            "branding": {
                "organization": organization.get("name", ""),
                "department": organization.get("department", ""),
                "logo_path": branding.get("logo_path", ""),
                "logo_alt": branding.get("logo_alt", organization.get("name", "")),
                "primary_color": colors.get("primary", ""),
            }
        },
    )
    template_defaults = _report_template_by_id(organization_profile, str(defaults.get("report_template", "")))
    if template_defaults:
        report_template = report.setdefault("template", {})
        if isinstance(report_template, dict):
            _merge_defaults(report_template, template_defaults)
            _merge_defaults(
                report_template,
                {
                    "company": organization.get("name", ""),
                    "prepared_by": organization.get("prepared_by", ""),
                    "checked_by": organization.get("checked_by", ""),
                    "approved_by": organization.get("approved_by", ""),
                },
            )
    post = out.setdefault("post", {})
    if not isinstance(post, dict):
        post = {}
        out["post"] = post
    _merge_defaults(
        post,
        {
            "style": {
                "palette": defaults.get("post_palette", "organization_safety"),
                "primary_color": colors.get("primary", ""),
                "accent_color": colors.get("accent", ""),
                "ok_color": colors.get("ok", ""),
                "warning_color": colors.get("warning", ""),
                "error_color": colors.get("error", ""),
            }
        },
    )
    out["organization_profile"] = {
        "profile_id": organization_profile.get("profile_id", "default_organization"),
        "revision": organization_profile.get("revision", ""),
        "organization": organization.get("name", ""),
        "template_id": selected_template,
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    return out


def write_customization_artifacts(
    output_dir: str | Path,
    *,
    profile: Mapping[str, Any] | None = None,
    template_id: str | None = None,
) -> dict[str, str]:
    """Write editable profile, template catalog, validation, and sample application artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    organization_profile = dict(profile or default_organization_profile())
    validation = validate_organization_profile(organization_profile)
    catalog_rows = project_template_catalog(organization_profile)
    profile_json = out / "organization_profile.json"
    profile_yaml = out / "organization_profile.yaml"
    validation_json = out / "customization_validation.json"
    catalog_json = out / "project_template_catalog.json"
    catalog_csv = out / "project_template_catalog.csv"
    catalog_html = out / "project_template_catalog.html"
    sample_yaml = out / "customized_sample_input.yaml"
    write_json_artifact(profile_json, organization_profile)
    profile_yaml.write_text(yaml.safe_dump(organization_profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json_artifact(validation_json, validation)
    write_json_artifact(catalog_json, {"schema": PROJECT_TEMPLATE_CATALOG_SCHEMA, "templates": catalog_rows})
    write_dict_rows_csv(catalog_csv, catalog_rows, ["id", "label", "domain", "solver_route", "unit_system", "report_template", "description"])
    write_html_artifact(
        catalog_html,
        html_table_document(
            title="GeoFEM project template catalog",
            headers=["id", "label", "domain", "solver", "unit", "report template", "description"],
            rows=[[row["id"], row["label"], row["domain"], row["solver_route"], row["unit_system"], row["report_template"], row["description"]] for row in catalog_rows],
            lead=f"Organization: {organization_profile.get('organization', {}).get('name', '') if isinstance(organization_profile.get('organization'), Mapping) else ''}",
        ),
    )
    from .samples import plane_strain_quad4_sample

    sample = apply_organization_profile(plane_strain_quad4_sample(), organization_profile, template_id=template_id)
    sample_yaml.write_text(yaml.safe_dump(sample, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {
        "profile_json": str(profile_json),
        "profile_yaml": str(profile_yaml),
        "validation": str(validation_json),
        "catalog_json": str(catalog_json),
        "catalog_csv": str(catalog_csv),
        "catalog_html": str(catalog_html),
        "customized_sample": str(sample_yaml),
    }


def _default_project_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": "geofem_review",
            "label": "GeoFEM 2D review template",
            "domain": "general_geotechnical",
            "solver_route": "GeoFEM 2D",
            "description": "Static plane-strain review with organization report defaults and Post palette.",
            "analysis_defaults": {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN"},
            "mesh_defaults": {"element_type": "QUAD4", "integration": "B-bar"},
            "solver_defaults": {"method": "direct"},
            "report_defaults": {"template": {"id": "organization_review_a4"}},
            "post_defaults": {"geofeas_style": True, "style": {"legend_position": "right", "relative_displacement": True}},
        },
        {
            "id": "geofeas_public_submission",
            "label": "GeoFEAS public substitute submission",
            "domain": "geo_feas_public_substitute",
            "solver_route": "GeoFEM 2D",
            "description": "Public-information-based GeoFEAS substitute workflow with calculation report bundle.",
            "analysis_defaults": {"dimension": "2D", "type": "static_plane_strain", "profile": "geofeas_public_v5", "unit_system": "m-kN"},
            "report_defaults": {"template": {"id": "client_submission_a3"}},
            "post_defaults": {"geofeas_style": True, "values_csv": True},
        },
        {
            "id": "vgflow_seepage",
            "label": "VGFlow2D seepage handoff template",
            "domain": "seepage",
            "solver_route": "VGFlow2D public substitute",
            "description": "Transient seepage settings and report defaults for VGFlow2D substitute handoff.",
            "analysis_defaults": {"dimension": "2D", "type": "vgflow2d", "mode": "transient", "unit_system": "m-kN"},
            "report_defaults": {"template": {"id": "organization_review_a4"}, "print_profile": {"paper": "A3", "orientation": "landscape"}},
            "vgflow2d_defaults": {"report": {"post_apply": ["post_contours", "flow_vectors"]}},
        },
    ]


def _apply_template_defaults(out: dict[str, Any], template: Mapping[str, Any]) -> None:
    for source_key, target_key in (
        ("analysis_defaults", "analysis"),
        ("mesh_defaults", "mesh"),
        ("solver_defaults", "solver"),
        ("report_defaults", "report"),
        ("post_defaults", "post"),
        ("vgflow2d_defaults", "vgflow2d"),
    ):
        defaults = _mapping(template.get(source_key))
        if not defaults:
            continue
        target = out.setdefault(target_key, {})
        if not isinstance(target, dict):
            target = {}
            out[target_key] = target
        _merge_defaults(target, defaults)


def _template_by_id(profile: Mapping[str, Any], template_id: str) -> Mapping[str, Any]:
    for row in _list_of_mappings(profile.get("project_templates")):
        if str(row.get("id", "")) == template_id:
            return row
    return {}


def _report_template_by_id(profile: Mapping[str, Any], template_id: str) -> Mapping[str, Any]:
    for row in _list_of_mappings(profile.get("report_templates")):
        if str(row.get("id", row.get("template_id", ""))) == template_id:
            return {key: value for key, value in row.items() if key != "id"}
    return {}


def _merge_defaults(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target or target[key] in (None, ""):
            target[key] = copy.deepcopy(value)
        elif isinstance(target.get(key), dict) and isinstance(value, Mapping):
            _merge_defaults(target[key], value)  # type: ignore[arg-type]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


__all__ = [
    "CUSTOMIZATION_VALIDATION_SCHEMA",
    "ORGANIZATION_PROFILE_SCHEMA",
    "PROJECT_TEMPLATE_CATALOG_SCHEMA",
    "apply_organization_profile",
    "default_organization_profile",
    "load_organization_profile",
    "project_template_catalog",
    "validate_organization_profile",
    "write_customization_artifacts",
]
