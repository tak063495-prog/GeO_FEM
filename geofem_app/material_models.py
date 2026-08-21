"""Material-model catalog, form schema, and inventory artifacts."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .fem2d_types import ElasticPlaneStrainMaterial


_BASE_FIELDS = [
    {"name": "E", "label": "Young modulus", "required": True, "default": None, "range": "E > 0", "aliases": ["young", "young_modulus"], "widget": "number"},
    {"name": "nu", "label": "Poisson ratio", "required": True, "default": 0.3, "range": "0 <= nu < 0.5", "aliases": ["poisson"], "widget": "number"},
    {"name": "gamma", "label": "Unit weight", "required": False, "default": 0.0, "range": "finite", "aliases": ["unit_weight"], "widget": "number"},
    {"name": "thickness", "label": "Thickness", "required": False, "default": 1.0, "range": "thickness > 0", "aliases": [], "widget": "number"},
    {"name": "k0", "label": "K0 coefficient", "required": False, "default": None, "range": "k0 >= 0", "aliases": ["K0"], "widget": "number"},
]


_MODEL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "elastic": {
        "label": "Linear elastic plane strain",
        "family": "linear",
        "aliases": ["linear_elastic", "plane_strain_elastic"],
        "fields": [],
        "internal_variables": [],
        "outputs": ["sigma_x", "sigma_y", "sigma_z", "tau_xy", "eps_x", "eps_y", "eps_z", "gamma_xy", "p", "q"],
        "notes": "Small-strain linear elastic material.",
    },
    "drucker_prager": {
        "label": "Drucker-Prager",
        "family": "plasticity",
        "aliases": ["dp"],
        "fields": [
            {"name": "cohesion", "required": False, "default": 0.0, "range": "cohesion >= 0", "aliases": ["c"], "widget": "number"},
            {"name": "friction_angle", "required": False, "default": 0.0, "range": "degrees or radians", "aliases": ["phi"], "widget": "number"},
            {"name": "dilation_angle", "required": False, "default": "phi", "range": "degrees or radians", "aliases": ["psi"], "widget": "number"},
            {"name": "hardening", "required": False, "default": 0.0, "range": "hardening >= 0", "aliases": ["H"], "widget": "number"},
        ],
        "internal_variables": ["plastic_strain", "kappa", "yield_value"],
        "outputs": ["plastic", "yield_value", "p", "q", "sigma_1", "sigma_3"],
        "notes": "Smooth pressure-dependent strength model.",
    },
    "mohr_coulomb": {
        "label": "Mohr-Coulomb",
        "family": "plasticity",
        "aliases": ["mc"],
        "fields": [
            {"name": "cohesion", "required": False, "default": 0.0, "range": "cohesion >= 0", "aliases": ["c"], "widget": "number"},
            {"name": "friction_angle", "required": False, "default": 0.0, "range": "degrees or radians", "aliases": ["phi"], "widget": "number"},
            {"name": "dilation_angle", "required": False, "default": "phi", "range": "degrees or radians", "aliases": ["psi"], "widget": "number"},
            {"name": "hardening", "required": False, "default": 0.0, "range": "hardening >= 0", "aliases": ["H"], "widget": "number"},
        ],
        "internal_variables": ["plastic_strain", "kappa", "active_set", "yield_value"],
        "outputs": ["plastic", "active_set", "yield_value", "p", "q", "sigma_1", "sigma_3"],
        "notes": "Principal-stress active-set return mapping.",
    },
    "von_mises": {
        "label": "von Mises / J2",
        "family": "plasticity",
        "aliases": ["j2"],
        "fields": [
            {"name": "yield_stress", "required": True, "default": None, "range": "yield_stress >= 0", "aliases": ["sigma_y", "sy", "yield"], "widget": "number"},
            {"name": "hardening", "required": False, "default": 0.0, "range": "hardening >= 0", "aliases": ["H"], "widget": "number"},
        ],
        "internal_variables": ["plastic_strain", "kappa", "yield_value"],
        "outputs": ["plastic", "yield_value", "q", "tau_max"],
        "notes": "Pressure-independent J2 plasticity.",
    },
    "no_tension": {
        "label": "Elastic with no-tension cutoff",
        "family": "tension_cutoff",
        "aliases": ["tension_cutoff"],
        "fields": [
            {"name": "tension_cutoff", "required": False, "default": True, "range": "boolean or mapping", "aliases": [], "widget": "checkbox"},
            {"name": "ft", "required": False, "default": 0.0, "range": "ft >= 0", "aliases": ["tensile_strength"], "widget": "number"},
            {"name": "tension_cutoff_stage", "required": False, "default": "corrector", "range": "predictor/corrector", "aliases": [], "widget": "choice"},
        ],
        "internal_variables": ["tension_clipped", "yield_value"],
        "outputs": ["plastic", "yield_value", "sigma_1"],
        "notes": "Tension cap applied to principal stresses.",
    },
    "nonlinear_elastic": {
        "label": "Nonlinear elastic",
        "family": "nonlinear_elastic",
        "aliases": ["hardin_drnevich", "hardin-drnevich", "hd", "duncan_chang", "ramberg_osgood"],
        "fields": [
            {"name": "G0", "required": False, "default": "E/(2(1+nu))", "range": "G0 > 0", "aliases": ["Gmax", "initial_shear_modulus"], "widget": "number"},
            {"name": "gamma_ref", "required": False, "default": 1.0e-3, "range": "gamma_ref > 0", "aliases": ["reference_strain", "gamma50"], "widget": "number"},
            {"name": "min_stiffness_ratio", "required": False, "default": 0.02, "range": "0..1", "aliases": ["Gmin_ratio"], "widget": "number"},
        ],
        "internal_variables": ["gamma_eq", "modulus_ratio", "effective_G", "effective_E"],
        "outputs": ["gamma_eq", "modulus_ratio", "effective_G", "effective_E"],
        "notes": "Strain-dependent equivalent shear stiffness.",
    },
    "uw_clay": {
        "label": "UW clay substitute",
        "family": "advanced_plasticity",
        "aliases": ["uw-clay"],
        "fields": [
            {"name": "su", "required": False, "default": "cohesion", "range": "su >= 0", "aliases": ["cu", "undrained_shear_strength"], "widget": "number"},
            {"name": "gamma_ref", "required": False, "default": 1.0e-3, "range": "gamma_ref > 0", "aliases": ["reference_strain"], "widget": "number"},
        ],
        "internal_variables": ["gamma_eq", "plastic_multiplier", "hardening_variable", "modulus_ratio"],
        "outputs": ["plastic", "hardening_variable", "modulus_ratio", "effective_G"],
        "notes": "Published-name substitute mapped to the native Drucker-Prager update.",
    },
    "pastor_zienkiewicz_sand": {
        "label": "Pastor-Zienkiewicz sand substitute",
        "family": "advanced_plasticity",
        "aliases": ["pastor-zienkiewicz-sand"],
        "fields": [
            {"name": "phi_cs", "required": False, "default": "friction_angle", "range": "degrees or radians", "aliases": ["critical_state_phi"], "widget": "number"},
            {"name": "peak_dilation_angle", "required": False, "default": "dilation_angle", "range": "degrees or radians", "aliases": ["psi_peak"], "widget": "number"},
            {"name": "gamma_ref", "required": False, "default": 1.0e-3, "range": "gamma_ref > 0", "aliases": ["reference_strain"], "widget": "number"},
        ],
        "internal_variables": ["gamma_eq", "dilatancy", "hardening_variable", "plastic_multiplier"],
        "outputs": ["plastic", "dilatancy", "hardening_variable", "modulus_ratio"],
        "notes": "Public substitute model using the native pressure-dependent plasticity core.",
    },
    "pastor_zienkiewicz_clay": {
        "label": "Pastor-Zienkiewicz clay substitute",
        "family": "advanced_plasticity",
        "aliases": ["pastor-zienkiewicz-clay"],
        "fields": [
            {"name": "su", "required": False, "default": "cohesion", "range": "su >= 0", "aliases": ["cu", "undrained_shear_strength"], "widget": "number"},
            {"name": "phi_cs", "required": False, "default": "friction_angle", "range": "degrees or radians", "aliases": ["critical_state_phi"], "widget": "number"},
            {"name": "gamma_ref", "required": False, "default": 1.0e-3, "range": "gamma_ref > 0", "aliases": ["reference_strain"], "widget": "number"},
        ],
        "internal_variables": ["gamma_eq", "dilatancy", "hardening_variable", "plastic_multiplier"],
        "outputs": ["plastic", "dilatancy", "hardening_variable", "modulus_ratio"],
        "notes": "Clay-oriented public substitute model.",
    },
    "liquefaction": {
        "label": "Liquefaction / ru-FL substitute",
        "family": "liquefaction",
        "aliases": ["bilinear_liquefaction", "bilinear-liquefaction"],
        "fields": [
            {"name": "cyclic_stress_ratio", "required": False, "default": 0.0, "range": "CSR >= 0", "aliases": ["CSR"], "widget": "number"},
            {"name": "cyclic_resistance_ratio", "required": False, "default": 0.0, "range": "CRR >= 0", "aliases": ["CRR", "RL20"], "widget": "number"},
            {"name": "generation_rate", "required": False, "default": 0.25, "range": ">= 0", "aliases": ["ru_generation_rate"], "widget": "number"},
            {"name": "dissipation_rate", "required": False, "default": 0.0, "range": ">= 0", "aliases": ["ru_dissipation_rate"], "widget": "number"},
            {"name": "post_liquefaction_stiffness_ratio", "required": False, "default": 0.02, "range": "0..1", "aliases": ["G_post_ratio"], "widget": "number"},
        ],
        "internal_variables": ["ru", "liquefaction_FL", "cycles", "cyclic_strain", "modulus_ratio"],
        "outputs": ["ru", "liquefaction_FL", "cycles", "cyclic_strain", "effective_G"],
        "notes": "Public-information substitute for cyclic ru generation and FL post-processing.",
    },
}


_ALIAS_TO_MODEL: dict[str, str] = {}
for _name, _definition in _MODEL_DEFINITIONS.items():
    _ALIAS_TO_MODEL[_name] = _name
    for _alias in _definition.get("aliases", []):
        _ALIAS_TO_MODEL[str(_alias).lower().replace("-", "_")] = _name


def normalize_material_model_name(model: Any, raw: Mapping[str, Any] | None = None) -> str:
    text = str(model or "elastic").lower().strip().replace("-", "_")
    if text in _ALIAS_TO_MODEL:
        return _ALIAS_TO_MODEL[text]
    if raw is not None and isinstance(raw.get("liquefaction"), Mapping):
        return "liquefaction"
    family = str((raw or {}).get("model_family", (raw or {}).get("family", ""))).lower().strip().replace("-", "_")
    if family in _ALIAS_TO_MODEL:
        return _ALIAS_TO_MODEL[family]
    return text


def material_model_definition(model: Any) -> dict[str, Any] | None:
    key = normalize_material_model_name(model)
    item = _MODEL_DEFINITIONS.get(key)
    if item is None:
        return None
    return {"name": key, **_copy_jsonable(item)}


def material_model_catalog() -> list[dict[str, Any]]:
    return [material_model_definition(name) for name in sorted(_MODEL_DEFINITIONS) if material_model_definition(name) is not None]  # type: ignore[list-item]


def material_form_schema(model: Any) -> dict[str, Any]:
    definition = material_model_definition(model)
    if definition is None:
        return {"model": str(model), "fields": list(_BASE_FIELDS), "unknown": True}
    fields = [_copy_jsonable(field) for field in _BASE_FIELDS]
    for field in definition.get("fields", []):
        fields.append(_copy_jsonable(field))
    return {
        "schema": "geofem.material_form_schema.v1",
        "model": definition["name"],
        "label": definition.get("label", definition["name"]),
        "family": definition.get("family", ""),
        "fields": fields,
        "internal_variables": list(definition.get("internal_variables", [])),
        "outputs": list(definition.get("outputs", [])),
    }


def material_required_parameters(model: Any) -> set[str]:
    return {str(field["name"]) for field in material_form_schema(model).get("fields", []) if bool(field.get("required", False))}


def material_validation_issues(name: str, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = normalize_material_model_name(raw.get("model", raw.get("type", "elastic")), raw)
    definition = material_model_definition(model)
    if definition is None:
        return [
            {
                "severity": "ERROR",
                "path": f"materials.{name}.model",
                "message": f"unsupported material model '{model}'",
                "suggestion": "Use one of: " + ", ".join(sorted(_MODEL_DEFINITIONS)),
            }
        ]
    provided = set(str(key) for key in raw)
    aliases: dict[str, set[str]] = {}
    for field in material_form_schema(model).get("fields", []):
        aliases[str(field["name"])] = {str(alias) for alias in field.get("aliases", [])}
    issues = []
    for required in material_required_parameters(model):
        if required in provided or provided.intersection(aliases.get(required, set())):
            continue
        issues.append(
            {
                "severity": "ERROR",
                "path": f"materials.{name}.{required}",
                "message": "required material parameter is missing",
                "suggestion": f"Set {required} for model {model}.",
            }
        )
    return issues


def build_material_inventory(materials: Mapping[str, Any], input_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw_materials: Mapping[str, Any] = {}
    if isinstance(input_config, Mapping):
        maybe = input_config.get("materials", input_config.get("material", {}))
        if isinstance(maybe, Mapping):
            raw_materials = maybe
    rows: list[dict[str, Any]] = []
    for name, material in sorted(materials.items(), key=lambda item: str(item[0])):
        raw = raw_materials.get(name, {})
        if isinstance(material, ElasticPlaneStrainMaterial):
            model = normalize_material_model_name(material.advanced_model or material.model)
            provided = set(raw) if isinstance(raw, Mapping) else set()
            missing = sorted(material_required_parameters(model) - provided) if provided else []
            row = {
                "name": str(name),
                "model": model,
                "family": (material_model_definition(model) or {}).get("family", ""),
                "E": material.E,
                "nu": material.nu,
                "gamma": material.gamma,
                "thickness": material.thickness,
                "provided_parameters": sorted(str(key) for key in provided),
                "missing_required": missing,
                "internal_variables": list((material_model_definition(model) or {}).get("internal_variables", [])),
                "outputs": list((material_model_definition(model) or {}).get("outputs", [])),
            }
        elif isinstance(material, Mapping):
            model = normalize_material_model_name(material.get("model", material.get("type", "elastic")), material)
            provided = set(str(key) for key in material)
            row = {
                "name": str(name),
                "model": model,
                "family": (material_model_definition(model) or {}).get("family", ""),
                "E": material.get("E", material.get("young", "")),
                "nu": material.get("nu", material.get("poisson", "")),
                "gamma": material.get("gamma", material.get("unit_weight", "")),
                "thickness": material.get("thickness", 1.0),
                "provided_parameters": sorted(provided),
                "missing_required": sorted(material_required_parameters(model) - provided),
                "internal_variables": list((material_model_definition(model) or {}).get("internal_variables", [])),
                "outputs": list((material_model_definition(model) or {}).get("outputs", [])),
            }
        else:
            continue
        rows.append(row)
    return {"schema": "geofem.material_inventory.v1", "material_count": len(rows), "materials": rows}


def write_material_reports(
    materials: Mapping[str, Any],
    output_dir: str | Path,
    *,
    input_config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = {"schema": "geofem.material_model_catalog.v1", "models": material_model_catalog()}
    inventory = build_material_inventory(materials, input_config)
    catalog_json = out / "material_model_catalog.json"
    inventory_json = out / "material_inventory.json"
    catalog_csv = out / "material_model_catalog.csv"
    inventory_csv = out / "material_inventory.csv"
    html_path = out / "material_models.html"
    catalog_json.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    inventory_json.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with catalog_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "label", "family", "required_parameters", "internal_variables", "outputs", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in catalog["models"]:
            schema = material_form_schema(row["name"])
            required = [field["name"] for field in schema.get("fields", []) if bool(field.get("required", False))]
            writer.writerow(
                {
                    "name": row.get("name", ""),
                    "label": row.get("label", ""),
                    "family": row.get("family", ""),
                    "required_parameters": ",".join(str(value) for value in required),
                    "internal_variables": ",".join(str(value) for value in row.get("internal_variables", [])),
                    "outputs": ",".join(str(value) for value in row.get("outputs", [])),
                    "notes": row.get("notes", ""),
                }
            )
    with inventory_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "model", "family", "E", "nu", "gamma", "thickness", "missing_required", "internal_variables", "outputs"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in inventory["materials"]:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
    html_path.write_text(_material_html(catalog, inventory), encoding="utf-8")
    return {
        "catalog_json": str(catalog_json),
        "catalog_csv": str(catalog_csv),
        "inventory_json": str(inventory_json),
        "inventory_csv": str(inventory_csv),
        "html": str(html_path),
    }


def _material_html(catalog: Mapping[str, Any], inventory: Mapping[str, Any]) -> str:
    catalog_rows = []
    for row in catalog.get("models", []):
        if not isinstance(row, Mapping):
            continue
        catalog_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{html.escape(str(row.get('label', '')))}</td>"
            f"<td>{html.escape(str(row.get('family', '')))}</td>"
            f"<td>{html.escape(','.join(str(v) for v in row.get('internal_variables', [])))}</td>"
            "</tr>"
        )
    inventory_rows = []
    for row in inventory.get("materials", []):
        if not isinstance(row, Mapping):
            continue
        inventory_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{html.escape(str(row.get('model', '')))}</td>"
            f"<td>{html.escape(str(row.get('E', '')))}</td>"
            f"<td>{html.escape(str(row.get('nu', '')))}</td>"
            f"<td>{html.escape(_csv_value(row.get('missing_required', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM material models</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>Material Models</h1>
<h2>Catalog</h2><table><thead><tr><th>model</th><th>label</th><th>family</th><th>internal variables</th></tr></thead><tbody>{''.join(catalog_rows)}</tbody></table>
<h2>Inventory</h2><table><thead><tr><th>material</th><th>model</th><th>E</th><th>nu</th><th>missing</th></tr></thead><tbody>{''.join(inventory_rows)}</tbody></table>
</body></html>
"""


def _copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


__all__ = [
    "normalize_material_model_name",
    "material_model_definition",
    "material_model_catalog",
    "material_form_schema",
    "material_required_parameters",
    "material_validation_issues",
    "build_material_inventory",
    "write_material_reports",
]
