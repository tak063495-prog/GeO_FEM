"""Workflow guidance helpers shared by the desktop GUI and tests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Mapping

from geofem_app.gui.i18n import DEFAULT_GUI_LOCALE, gui_message
from geofem_app.material_models import material_validation_issues


@dataclass(frozen=True)
class WorkflowStepSpec:
    id: str
    label: str
    panel: str
    action_group: str
    required: bool
    next_action: str


WORKFLOW_STEPS: tuple[WorkflowStepSpec, ...] = (
    WorkflowStepSpec("analysis", "解析条件", "analysis", "primary", True, "解析種別、次元、単位系を確認します。"),
    WorkflowStepSpec("geometry", "形状/CAD", "geometry", "primary", False, "形状線、CAD、またはメッシュ生成条件を確認します。"),
    WorkflowStepSpec("mesh", "メッシュ", "mesh", "primary", True, "メッシュ生成条件、要素種別、分割数を設定します。"),
    WorkflowStepSpec("materials", "材料", "materials", "primary", True, "材料モデルと主要パラメータを設定します。"),
    WorkflowStepSpec("boundary_conditions", "境界条件", "boundary_conditions", "primary", True, "拘束条件またはステージ内境界条件を設定します。"),
    WorkflowStepSpec("loads", "荷重", "loads", "detail", False, "荷重、自己重量、水圧、ステージ荷重を必要に応じて設定します。"),
    WorkflowStepSpec("stages", "ステージ", "stages", "detail", False, "施工段階、材料変更、応力解放率を必要に応じて設定します。"),
    WorkflowStepSpec("model_check", "モデルチェック", "model_check", "confirm", True, "解析前チェックを実行し、エラーを解消します。"),
    WorkflowStepSpec("solver", "解析実行", "solver", "confirm", True, "解析を実行し、収束条件と出力先を確認します。"),
    WorkflowStepSpec("results", "結果確認", "results", "output", False, "変位、応力、FL、履歴などの結果を確認します。"),
    WorkflowStepSpec("report", "帳票", "report", "output", False, "HTML/PDF帳票とmanifestを確認します。"),
)


ACTION_GROUP_LABELS = {
    "primary": "主要操作",
    "confirm": "確認",
    "output": "出力",
    "detail": "詳細設定",
}


def workflow_steps(*, locale: str = DEFAULT_GUI_LOCALE) -> list[dict[str, Any]]:
    """Return the stable workflow definition exposed to the GUI."""

    return [
        {
            "id": step.id,
            "label": _step_message(step.id, "label", step.label, locale),
            "panel": step.panel,
            "action_group": step.action_group,
            "action_group_label": _action_group_label(step.action_group, locale),
            "required": step.required,
            "next_action": _step_message(step.id, "next", step.next_action, locale),
        }
        for step in WORKFLOW_STEPS
    ]


def build_workflow_guidance(
    cfg: Mapping[str, Any],
    *,
    result_dir: str | Path | None = None,
    locale: str = DEFAULT_GUI_LOCALE,
) -> dict[str, Any]:
    """Build a serializable workflow completion summary for GUI navigation."""

    result_path = Path(result_dir) if result_dir is not None else None
    step_rows: list[dict[str, Any]] = []
    for spec in WORKFLOW_STEPS:
        completed, missing_paths, detail = _evaluate_step(spec.id, cfg, result_path)
        step_rows.append(
            {
                "id": spec.id,
                "label": _step_message(spec.id, "label", spec.label, locale),
                "panel": spec.panel,
                "action_group": spec.action_group,
                "action_group_label": _action_group_label(spec.action_group, locale),
                "required": spec.required,
                "completed": bool(completed),
                "status": "complete" if completed else ("missing" if spec.required else "optional"),
                "status_label": _status_label("complete" if completed else ("missing" if spec.required else "optional"), locale),
                "completion_ratio": 1.0 if completed else 0.0,
                "missing_paths": missing_paths,
                "next_action": gui_message("workflow.completed", locale=locale) if completed else _step_message(spec.id, "next", spec.next_action, locale),
                "jump_target": spec.panel,
                "detail": _step_message(spec.id, "detail", detail, locale),
            }
        )

    required_rows = [row for row in step_rows if row["required"]]
    completed_count = sum(1 for row in required_rows if row["completed"])
    completion_ratio = completed_count / len(required_rows) if required_rows else 1.0
    next_step = next((row for row in step_rows if row["required"] and not row["completed"]), None)
    if next_step is None:
        next_step = next((row for row in step_rows if not row["completed"]), None)
    action_groups = _build_action_groups(step_rows)

    return {
        "schema": "geofem.gui.workflow_guidance.v1",
        "locale": locale,
        "passed": all(row["completed"] for row in required_rows),
        "completion_ratio": completion_ratio,
        "completed_required_count": completed_count,
        "required_count": len(required_rows),
        "missing_required_count": len(required_rows) - completed_count,
        "next_step": _compact_step(next_step),
        "steps": step_rows,
        "action_groups": action_groups,
        "features": {
            "workflow_navigation": True,
            "next_action": True,
            "required_input_completion": True,
            "missing_input_jump_targets": True,
            "action_hierarchy": True,
        },
    }


def write_workflow_guidance(
    cfg: Mapping[str, Any],
    output_dir: str | Path,
    *,
    result_dir: str | Path | None = None,
    locale: str = DEFAULT_GUI_LOCALE,
) -> dict[str, str]:
    """Write JSON/CSV/HTML workflow guidance artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    guidance = build_workflow_guidance(cfg, result_dir=result_dir, locale=locale)
    json_path = out / "workflow_guidance.json"
    csv_path = out / "workflow_guidance.csv"
    html_path = out / "workflow_guidance.html"

    json_path.write_text(json.dumps(guidance, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "label",
                "action_group_label",
                "required",
                "completed",
                "status",
                "missing_paths",
                "next_action",
                "jump_target",
            ],
        )
        writer.writeheader()
        for row in guidance["steps"]:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in writer.fieldnames})
    html_path.write_text(_html_report(guidance, locale=locale), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _evaluate_step(step_id: str, cfg: Mapping[str, Any], result_dir: Path | None) -> tuple[bool, list[str], str]:
    if step_id == "analysis":
        analysis = _mapping(cfg.get("analysis"))
        missing = []
        if not analysis:
            missing.append("analysis")
        if str(analysis.get("dimension", "2D")).upper() != "2D":
            missing.append("analysis.dimension")
        if not analysis.get("type"):
            missing.append("analysis.type")
        return not missing, missing, "解析種別と2D指定を確認します。"
    if step_id == "geometry":
        geometry = _mapping(cfg.get("geometry"))
        mesh = _mapping(cfg.get("mesh"))
        has_geometry = any(_has_items(geometry.get(key)) for key in ("lines", "regions", "tunnels", "blocks", "cad_lines"))
        completed = has_geometry or bool(mesh.get("generator")) or _has_items(mesh.get("elements"))
        return completed, [] if completed else ["geometry.lines", "mesh.generator"], "形状定義またはメッシュ生成条件があれば先へ進めます。"
    if step_id == "mesh":
        mesh = _mapping(cfg.get("mesh"))
        missing = []
        if not mesh:
            missing.append("mesh")
        if bool(mesh.get("requires_rebuild", False)):
            missing.append("mesh.rebuild_required")
        generator = str(mesh.get("generator", "")).lower()
        if generator == "rectangle":
            if not _positive_number(mesh.get("nx")):
                missing.append("mesh.nx")
            if not _positive_number(mesh.get("ny")):
                missing.append("mesh.ny")
        elif not (_has_items(mesh.get("nodes")) and _has_items(mesh.get("elements"))):
            missing.append("mesh.generator_or_nodes_elements")
        if not _mesh_has_element_type(mesh):
            missing.append("mesh.element_type")
        return not missing, missing, "形状変更後はメッシュを再構成し、要素種別と分割条件を確認します。"
    if step_id == "materials":
        materials = _mapping(cfg.get("materials"))
        missing = []
        if not materials:
            missing.append("materials")
        for name, raw in materials.items():
            material = _mapping(raw)
            if not material.get("model") and not material.get("type"):
                missing.append(f"materials.{name}.model")
            if not _positive_number(material.get("E", material.get("young"))):
                missing.append(f"materials.{name}.E")
            if material.get("nu", material.get("poisson")) is None:
                missing.append(f"materials.{name}.nu")
            for issue in material_validation_issues(str(name), material):
                path = str(issue.get("path", ""))
                if path:
                    missing.append(path)
        missing.extend(_material_assignment_missing_paths(cfg, materials))
        return not missing, missing, "材料モデル、E、nuなどを確認します。"
    if step_id == "boundary_conditions":
        completed = _has_items(cfg.get("boundary_conditions")) or _stage_has_items(cfg, "boundary_conditions") or _stage_has_items(cfg, "bc")
        return completed, [] if completed else ["boundary_conditions"], "拘束条件を設定します。"
    if step_id == "loads":
        completed = _has_items(cfg.get("loads")) or _stage_has_items(cfg, "loads") or _stage_has_items(cfg, "hydro") or _stage_has_items(cfg, "water_pressure")
        return completed, [] if completed else ["loads"], "荷重が不要な解析では省略できます。"
    if step_id == "stages":
        completed = _has_items(cfg.get("stages")) or _has_items(cfg.get("steps"))
        return completed, [] if completed else ["stages"], "単一ステージ解析では省略できます。"
    if step_id == "model_check":
        required_ready = _required_precheck_ready(cfg)
        return required_ready, [] if required_ready else ["analysis", "mesh", "materials", "boundary_conditions"], "解析前チェックを実行します。"
    if step_id == "solver":
        required_ready = _required_precheck_ready(cfg)
        return required_ready, [] if required_ready else ["model_check"], "モデルチェック後に解析を実行します。"
    if step_id == "results":
        completed = _exists(result_dir, "summary.json") or _exists(result_dir, "failure_report.json")
        return completed, [] if completed else ["results/summary.json"], "解析後に結果ファイルを確認します。"
    if step_id == "report":
        completed = any(_exists(result_dir, name) for name in ("standard_report.html", "calculation_report.html", "report_manifest.json"))
        return completed, [] if completed else ["results/standard_report.html"], "帳票生成後にHTML/PDFとmanifestを確認します。"
    return False, [step_id], ""


def _required_precheck_ready(cfg: Mapping[str, Any]) -> bool:
    for step_id in ("analysis", "mesh", "materials", "boundary_conditions"):
        completed, _missing, _detail = _evaluate_step(step_id, cfg, None)
        if not completed:
            return False
    return True


def _build_action_groups(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in ACTION_GROUP_LABELS}
    for row in steps:
        groups[row["action_group"]].append(
            {
                "id": row["id"],
                "label": row["label"],
                "target_panel": row["panel"],
                "completed": row["completed"],
                "next_action": row["next_action"],
            }
        )
    return groups


def _compact_step(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": row["id"],
        "label": row["label"],
        "panel": row["panel"],
        "required": row["required"],
        "missing_paths": list(row["missing_paths"]),
        "next_action": row["next_action"],
    }


def _html_report(guidance: Mapping[str, Any], *, locale: str = DEFAULT_GUI_LOCALE) -> str:
    rows = []
    for row in guidance["steps"]:
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['label']))}</td>"
            f"<td>{escape(str(row['action_group_label']))}</td>"
            f"<td>{escape(str(row.get('status_label', _status_label(str(row.get('status', 'optional')), locale))))}</td>"
            f"<td>{escape(', '.join(row['missing_paths']))}</td>"
            f"<td>{escape(str(row['next_action']))}</td>"
            f"<td>{escape(str(row['jump_target']))}</td>"
            "</tr>"
        )
    ratio = float(guidance["completion_ratio"]) * 100.0
    return (
        f"<!doctype html><html lang=\"{escape(locale)}\"><meta charset=\"utf-8\">"
        "<title>GeoFEM workflow guidance</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}th{background:#f4f4f4}</style>"
        f"<h1>{escape(gui_message('workflow.report.title', locale=locale))}</h1>"
        f"<p>{escape(gui_message('workflow.progress_ratio', locale=locale, ratio=ratio))}</p>"
        "<table><thead><tr>"
        f"<th>{escape(gui_message('workflow.table.step', locale=locale))}</th>"
        f"<th>{escape(gui_message('workflow.table.role', locale=locale))}</th>"
        f"<th>{escape(gui_message('workflow.table.status', locale=locale))}</th>"
        f"<th>{escape(gui_message('workflow.table.missing', locale=locale))}</th>"
        f"<th>{escape(gui_message('workflow.table.next', locale=locale))}</th>"
        f"<th>{escape(gui_message('workflow.table.jump', locale=locale))}</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></html>"
    )


def _step_message(step_id: str, suffix: str, fallback: str, locale: str) -> str:
    key = f"workflow.step.{step_id}.{suffix}"
    value = gui_message(key, locale=locale)
    return fallback if value == key else value


def _action_group_label(group: str, locale: str) -> str:
    key = f"gui.action_role.{group}"
    value = gui_message(key, locale=locale)
    return ACTION_GROUP_LABELS.get(group, group) if value == key else value


def _status_label(status: str, locale: str) -> str:
    key = f"workflow.status.{status}"
    value = gui_message(key, locale=locale)
    return status if value == key else value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_items(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return value is not None and value != ""


def _mesh_has_element_type(mesh: Mapping[str, Any]) -> bool:
    if str(mesh.get("element_type", mesh.get("type", ""))).strip():
        return True
    elements = mesh.get("elements", [])
    if isinstance(elements, Mapping):
        iterable = elements.values()
    elif isinstance(elements, (list, tuple)):
        iterable = elements
    else:
        return False
    has_element = False
    for element in iterable:
        if not isinstance(element, Mapping):
            return False
        has_element = True
        if not str(element.get("type", element.get("element_type", ""))).strip():
            return False
    return has_element


def _stage_has_items(cfg: Mapping[str, Any], key: str) -> bool:
    for stage_key in ("stages", "steps"):
        stages = cfg.get(stage_key)
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if isinstance(stage, Mapping) and _has_items(stage.get(key)):
                return True
    return False


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _region_setting_key_candidates(region_key: str, region_ids: list[str]) -> list[str]:
    region_key = str(region_key)
    candidates = [region_key]
    if region_key in region_ids:
        candidates.append(f"region_{region_ids.index(region_key) + 1}")
    elif region_key.startswith("region_"):
        suffix = region_key.removeprefix("region_")
        if suffix.isdigit():
            index = int(suffix) - 1
            if 0 <= index < len(region_ids):
                candidates.append(region_ids[index])
        else:
            candidates.append(suffix)
    elif region_key.isdigit():
        candidates.append(f"region_{region_key}")
        index = int(region_key) - 1
        if 0 <= index < len(region_ids):
            candidates.append(region_ids[index])
    return list(dict.fromkeys(candidates))


def _material_assignment_missing_paths(cfg: Mapping[str, Any], materials: Mapping[str, Any]) -> list[str]:
    geometry = _mapping(cfg.get("geometry"))
    raw_regions = geometry.get("regions", [])
    if not isinstance(raw_regions, list) or not raw_regions:
        return []
    mesh = _mapping(cfg.get("mesh"))
    region_settings = _mapping(mesh.get("region_settings"))
    region_ids = [
        str(raw.get("id", f"region_{index}") or f"region_{index}") if isinstance(raw, Mapping) else f"region_{index}"
        for index, raw in enumerate(raw_regions, start=1)
    ]
    missing: list[str] = []
    for index, raw in enumerate(raw_regions, start=1):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("source", "")).strip() == "mesh_rectangle_outline":
            continue
        region_id = str(raw.get("id", f"region_{index}") or f"region_{index}")
        material = str(raw.get("material", raw.get("mat", "")) or "").strip()
        if not material:
            for candidate in _region_setting_key_candidates(region_id, region_ids):
                setting = region_settings.get(candidate)
                if isinstance(setting, Mapping):
                    material = str(setting.get("material", "") or "").strip()
                    if material:
                        break
        if not material:
            missing.append(f"geometry.regions.{region_id}.material")
        elif material not in materials:
            missing.append(f"materials.{material}")
    return missing


def _exists(root: Path | None, relative: str) -> bool:
    return root is not None and (root / relative).exists()


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)
