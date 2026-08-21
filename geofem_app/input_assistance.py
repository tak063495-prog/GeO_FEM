"""Input templates, unit hints, and range diagnostics for GeoFEM configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .material_models import material_form_schema, normalize_material_model_name
from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


SUPPORTED_UNIT_SYSTEMS = {
    "m-kn": {"length": "m", "force": "kN", "stress": "kPa", "unit_weight": "kN/m3", "displacement": "m"},
    "m-n": {"length": "m", "force": "N", "stress": "Pa", "unit_weight": "N/m3", "displacement": "m"},
    "si": {"length": "m", "force": "N", "stress": "Pa", "unit_weight": "N/m3", "displacement": "m"},
    "mm-n": {"length": "mm", "force": "N", "stress": "MPa", "unit_weight": "N/mm3", "displacement": "mm"},
    "user": {"length": "user", "force": "user", "stress": "user", "unit_weight": "user", "displacement": "user"},
    "custom": {"length": "custom", "force": "custom", "stress": "custom", "unit_weight": "custom", "displacement": "custom"},
}


INPUT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "analysis": [
        {
            "id": "static_plane_strain",
            "label": "静的平面ひずみ",
            "snippet": {"analysis": {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN", "fields": ["u"]}},
        },
        {
            "id": "axisymmetric_static",
            "label": "軸対称静的",
            "snippet": {"analysis": {"dimension": "2D", "type": "axisymmetric_static", "geometry": "axisymmetric", "unit_system": "m-kN", "fields": ["u"]}},
        },
        {
            "id": "consolidation_up",
            "label": "u-p 圧密",
            "snippet": {"analysis": {"dimension": "2D", "type": "consolidation", "unit_system": "m-kN", "fields": ["u", "p"]}},
        },
    ],
    "material": [
        {
            "id": "elastic_soil",
            "label": "弾性地盤",
            "snippet": {"materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.3, "gamma": 18.0}}},
        },
        {
            "id": "mohr_coulomb_soil",
            "label": "Mohr-Coulomb 地盤",
            "snippet": {"materials": {"soil": {"model": "mohr_coulomb", "E": 20000.0, "nu": 0.3, "cohesion": 10.0, "friction_angle": 30.0, "gamma": 18.0}}},
        },
        {
            "id": "liquefaction_sand",
            "label": "液状化砂質土",
            "snippet": {"materials": {"sand": {"model": "liquefaction", "E": 30000.0, "nu": 0.33, "cyclic_stress_ratio": 0.18, "cyclic_resistance_ratio": 0.25}}},
        },
    ],
    "boundary": [
        {
            "id": "fixed_bottom",
            "label": "底面固定",
            "snippet": {"boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}]},
        },
        {
            "id": "side_roller",
            "label": "側方ローラー",
            "snippet": {"boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "right", "ux": 0.0}]},
        },
        {
            "id": "stage_hydrostatic_pressure",
            "label": "ステージ水圧",
            "snippet": {"stages": [{"name": "water", "type": "static", "hydro": {"pressure_bcs": [{"set": "waterline", "value": 0.0}]}}]},
        },
    ],
}


def input_assistance_template_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for area, templates in INPUT_TEMPLATES.items():
        for template in templates:
            rows.append({"area": area, **dict(template)})
    return rows


def build_input_assistance_summary(cfg: Mapping[str, Any], *, locale: str = "ja") -> dict[str, Any]:
    """Build unit/range guidance rows and immediate diagnostics for input forms."""

    unit_system = _unit_system(cfg)
    units = SUPPORTED_UNIT_SYSTEMS.get(unit_system, SUPPORTED_UNIT_SYSTEMS["m-kn"])
    guidance: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    _analysis_guidance(cfg, units, guidance, diagnostics)
    _material_guidance(cfg, units, guidance, diagnostics)
    _boundary_guidance(cfg, units, guidance, diagnostics)
    _solver_guidance(cfg, units, guidance, diagnostics)

    errors = sum(1 for row in diagnostics if row["severity"] == "ERROR")
    warnings = sum(1 for row in diagnostics if row["severity"] == "WARN")
    return {
        "schema": "geofem.input_assistance.v1",
        "locale": locale,
        "unit_system": unit_system,
        "units": units,
        "passed": errors == 0,
        "error_count": errors,
        "warning_count": warnings,
        "guidance_count": len(guidance),
        "diagnostic_count": len(diagnostics),
        "features": [
            "analysis_templates",
            "material_templates",
            "boundary_templates",
            "field_units",
            "recommended_ranges",
            "prohibited_ranges",
            "immediate_config_diagnostics",
        ],
        "templates": input_assistance_template_catalog(),
        "guidance_rows": guidance,
        "diagnostics": diagnostics,
    }


def write_input_assistance_artifacts(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out / "input_assistance.json"),
        "csv": str(out / "input_assistance.csv"),
        "html": str(out / "input_assistance.html"),
    }
    rows = [row for row in summary.get("guidance_rows", []) if isinstance(row, Mapping)]
    write_json_artifact(paths["json"], summary)
    write_dict_rows_csv(paths["csv"], rows, ["area", "path", "field", "unit", "recommended", "prohibited", "status", "message", "template_id"])
    write_html_artifact(paths["html"], _assistance_html(summary))
    return paths


def _analysis_guidance(cfg: Mapping[str, Any], units: Mapping[str, str], guidance: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    analysis = _mapping(cfg.get("analysis", {}))
    unit_system = _unit_system(cfg)
    _add_guidance(guidance, "analysis", "analysis.unit_system", "unit_system", "", "m-kN / m-N / SI / mm-N / user / custom", "unsupported unit-system text", "OK", f"current={unit_system}", "static_plane_strain")
    if unit_system not in SUPPORTED_UNIT_SYSTEMS:
        _add_diag(diagnostics, "WARN", "analysis.unit_system", f"未定義の単位系です: {unit_system}", "m-kN、m-N、SI、mm-N、user、custom のいずれかを指定してください。")
    analysis_type = str(analysis.get("type", "static_plane_strain"))
    _add_guidance(guidance, "analysis", "analysis.type", "type", "", "static_plane_strain / axisymmetric_static / consolidation / dynamic", "empty or unsupported type", "OK", f"current={analysis_type}", "static_plane_strain")
    dimension = str(analysis.get("dimension", "2D")).upper()
    _add_guidance(guidance, "analysis", "analysis.dimension", "dimension", "", "2D", "3D", "OK" if dimension == "2D" else "ERROR", f"current={dimension}", "static_plane_strain")
    if dimension != "2D":
        _add_diag(diagnostics, "ERROR", "analysis.dimension", "GeoFEM GUI は2D専用です。", "analysis.dimension を 2D にしてください。")
    _add_guidance(guidance, "analysis", "loads.*.fx/fy", "nodal force", units.get("force", ""), "load case and sign convention should be explicit", "non-finite values", "INFO", "節点荷重の単位を確認してください。", "")


def _material_guidance(cfg: Mapping[str, Any], units: Mapping[str, str], guidance: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    materials = _mapping(cfg.get("materials", cfg.get("material", {})))
    if not materials:
        _add_diag(diagnostics, "ERROR", "materials", "材料テンプレートが未設定です。", "elastic_soil または mohr_coulomb_soil テンプレートから材料を追加してください。")
        _add_guidance(guidance, "material", "materials", "template", "", "elastic_soil / mohr_coulomb_soil / liquefaction_sand", "empty materials", "ERROR", "材料定義がありません。", "elastic_soil")
        return
    for name, raw in materials.items():
        mat = _mapping(raw)
        model = normalize_material_model_name(mat.get("model", mat.get("type", "elastic")), mat)
        template_id = "mohr_coulomb_soil" if model == "mohr_coulomb" else ("liquefaction_sand" if model == "liquefaction" else "elastic_soil")
        for field in material_form_schema(model).get("fields", []):
            field_name = str(field.get("name", ""))
            unit = _material_unit(field_name, units)
            recommended, prohibited = _material_range_text(field_name, field)
            value = mat.get(field_name)
            status, message = _range_status(field_name, value)
            _add_guidance(guidance, "material", f"materials.{name}.{field_name}", field_name, unit, recommended, prohibited, status, message, template_id)
            if status in {"ERROR", "WARN"}:
                _add_diag(diagnostics, status, f"materials.{name}.{field_name}", message, f"{field_name} は {recommended} を目安にし、{prohibited} を避けてください。")


def _boundary_guidance(cfg: Mapping[str, Any], units: Mapping[str, str], guidance: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    bcs = _ensure_list(cfg.get("boundary_conditions", cfg.get("bc", [])))
    if not bcs:
        _add_diag(diagnostics, "WARN", "boundary_conditions", "変位拘束テンプレートが未設定です。", "fixed_bottom と side_roller などで剛体運動を止めてください。")
        _add_guidance(guidance, "boundary", "boundary_conditions", "template", "", "fixed_bottom / side_roller", "no displacement constraints", "WARN", "境界条件がありません。", "fixed_bottom")
    for index, raw in enumerate(bcs):
        bc = _mapping(raw)
        target_ok = any(key in bc for key in ("set", "node", "nodes", "edge", "edges"))
        comp_keys = [key for key in ("ux", "uy", "x", "y", "fixed") if key in bc]
        status = "OK" if target_ok and comp_keys else "WARN"
        if not target_ok or not comp_keys:
            _add_diag(diagnostics, "WARN", f"boundary_conditions[{index}]", "境界条件の対象または成分が不足しています。", "set/node と ux/uy または fixed を指定してください。")
        _add_guidance(guidance, "boundary", f"boundary_conditions[{index}]", "ux/uy", units.get("displacement", ""), "0.0 for support, prescribed displacement for loading", "missing target or component", status, f"components={','.join(comp_keys) or 'none'}", "fixed_bottom")
    for index, raw in enumerate(_ensure_list(cfg.get("loads", []))):
        load = _mapping(raw)
        if not load:
            continue
        load_type = str(load.get("type", "")).lower().strip()
        is_body_load = load_type in {"gravity", "self_weight", "body"} or bool(load.get("self_weight", False))
        body_fields = [key for key in ("gx", "gy", "bx", "by", "body_fx", "body_fy", "body_x", "body_y", "scale") if key in load]
        load_fields = [key for key in ("fx", "fy", "tx", "ty", "px", "py") if key in load]
        if is_body_load:
            load_fields.extend(body_fields or [load_type or "self_weight"])
        unit = units.get("force", "") if any(key in load for key in ("fx", "fy")) else units.get("stress", "")
        status = "OK" if load_fields else "WARN"
        if not load_fields:
            _add_diag(diagnostics, "WARN", f"loads[{index}]", "荷重成分が不足しています。", "fx/fy または tx/ty などの荷重成分を指定してください。")
        _add_guidance(guidance, "boundary", f"loads[{index}]", "load", unit, "explicit load case and sign convention", "missing load component", status, f"components={','.join(load_fields) or 'none'}", "")


def _solver_guidance(cfg: Mapping[str, Any], units: Mapping[str, str], guidance: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    _ = units
    solver = _mapping(cfg.get("solver", {}))
    tolerance = solver.get("tolerance", solver.get("tol", None))
    status, message = _generic_positive_status(tolerance, default_message="未指定時は既定の収束許容値を使用します。")
    _add_guidance(guidance, "analysis", "solver.tolerance", "tolerance", "", "1e-10 to 1e-5", "<= 0", status, message, "")
    if status == "ERROR":
        _add_diag(diagnostics, "ERROR", "solver.tolerance", message, "正の収束許容値を指定してください。")


def _material_unit(field: str, units: Mapping[str, str]) -> str:
    if field in {"E", "G0", "cohesion", "yield_stress", "hardening", "su", "ft"}:
        return units.get("stress", "")
    if field == "gamma":
        return units.get("unit_weight", "")
    if "angle" in field or field in {"nu", "k0", "min_stiffness_ratio", "post_liquefaction_stiffness_ratio", "cyclic_stress_ratio", "cyclic_resistance_ratio"}:
        return "-"
    if "strain" in field:
        return "-"
    return ""


def _material_range_text(field: str, schema_field: Mapping[str, Any]) -> tuple[str, str]:
    defaults = {
        "E": ("E > 0; typical soil values should match unit system", "<= 0"),
        "nu": ("0 <= nu < 0.5", "nu < 0 or nu >= 0.5"),
        "gamma": ("0 to 30 for ordinary soil unit weight in kN/m3 class units", "< 0"),
        "cohesion": (">= 0", "< 0"),
        "friction_angle": ("0 to 60 deg for ordinary soil input", "< 0 or very large angle"),
        "dilation_angle": ("0 to friction angle for common soil input", "very large angle"),
    }
    if field in defaults:
        return defaults[field]
    text = str(schema_field.get("range", "finite"))
    return text, "non-finite or outside model range"


def _range_status(field: str, value: Any) -> tuple[str, str]:
    if value is None:
        return "INFO", "未入力時は既定値またはモデル既定を使用します。"
    number = _float_or_none(value)
    if field in {"E", "thickness", "G0", "gamma_ref"}:
        if number is not None and number <= 0.0:
            return "ERROR", f"{field} は正の値が必要です。"
    if field in {"cohesion", "yield_stress", "hardening", "su", "ft", "gamma", "k0", "cyclic_stress_ratio", "cyclic_resistance_ratio", "generation_rate", "dissipation_rate"}:
        if number is not None and number < 0.0:
            return "ERROR", f"{field} は負値にできません。"
    if field == "nu" and number is not None and not 0.0 <= number < 0.5:
        return "ERROR", "nu は 0 以上 0.5 未満で指定してください。"
    if field in {"friction_angle", "dilation_angle", "phi_cs", "peak_dilation_angle"} and number is not None and (number < 0.0 or number > 75.0):
        return "WARN", f"{field} が一般的な角度範囲から外れています。単位がdeg/radのどちらか確認してください。"
    if field in {"min_stiffness_ratio", "post_liquefaction_stiffness_ratio"} and number is not None and not 0.0 <= number <= 1.0:
        return "ERROR", f"{field} は 0 から 1 の範囲で指定してください。"
    return "OK", "入力範囲を確認しました。"


def _generic_positive_status(value: Any, *, default_message: str) -> tuple[str, str]:
    if value is None:
        return "INFO", default_message
    number = _float_or_none(value)
    if number is None:
        return "WARN", "数値として解釈できません。"
    if number <= 0.0:
        return "ERROR", "正の値が必要です。"
    return "OK", "入力範囲を確認しました。"


def _add_guidance(
    rows: list[dict[str, Any]],
    area: str,
    path: str,
    field: str,
    unit: str,
    recommended: str,
    prohibited: str,
    status: str,
    message: str,
    template_id: str,
) -> None:
    rows.append(
        {
            "area": area,
            "path": path,
            "field": field,
            "unit": unit,
            "recommended": recommended,
            "prohibited": prohibited,
            "status": status,
            "message": message,
            "template_id": template_id,
        }
    )


def _add_diag(rows: list[dict[str, Any]], severity: str, path: str, message: str, suggestion: str) -> None:
    rows.append({"severity": severity, "path": path, "message": message, "suggestion": suggestion})


def _assistance_html(summary: Mapping[str, Any]) -> str:
    rows = [
        [
            row.get("area", ""),
            row.get("path", ""),
            row.get("field", ""),
            row.get("unit", ""),
            row.get("recommended", ""),
            row.get("prohibited", ""),
            row.get("status", ""),
            row.get("message", ""),
            row.get("template_id", ""),
        ]
        for row in summary.get("guidance_rows", [])
        if isinstance(row, Mapping)
    ]
    return html_table_document(
        title="GeoFEM 入力補助",
        lead=f"unit_system={summary.get('unit_system', '')}, errors={summary.get('error_count', 0)}, warnings={summary.get('warning_count', 0)}",
        headers=["area", "path", "field", "unit", "recommended", "prohibited", "status", "message", "template_id"],
        rows=rows,
    )


def _unit_system(cfg: Mapping[str, Any]) -> str:
    analysis = _mapping(cfg.get("analysis", {}))
    return str(analysis.get("unit_system", analysis.get("units", "m-kN"))).lower().replace(" ", "")


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


__all__ = [
    "INPUT_TEMPLATES",
    "SUPPORTED_UNIT_SYSTEMS",
    "build_input_assistance_summary",
    "input_assistance_template_catalog",
    "write_input_assistance_artifacts",
]
