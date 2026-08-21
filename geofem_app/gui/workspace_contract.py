"""Workspace responsibility contracts for the desktop GUI.

These contracts are intentionally UI-framework independent so tests, docs, and
the PySide window can share the same responsibility boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geofem_app.gui.i18n import DEFAULT_GUI_LOCALE, gui_message
from geofem_app.gui.workflow_guidance import workflow_steps


WORKSPACE_CONTRACT_SCHEMA = "geofem.gui.workspace_responsibility.v1"
WORKSPACE_CONTRACT_VALIDATION_SCHEMA = "geofem.gui.workspace_responsibility_validation.v1"


@dataclass(frozen=True)
class WorkspaceAreaSpec:
    id: str
    label_ja: str
    label_en: str
    role: str
    allowed: tuple[str, ...]
    disallowed: tuple[str, ...]


WORKSPACE_AREAS: tuple[WorkspaceAreaSpec, ...] = (
    WorkspaceAreaSpec(
        "left_tree",
        "左ツリー",
        "Left Tree",
        "navigation",
        (
            "作業場所の選択",
            "結果図へのナビゲーション",
            "帳票確認へのナビゲーション",
            "設定項目へのナビゲーション",
            "入力課題の場所表示",
        ),
        ("詳細入力フォーム", "解析実行", "帳票生成", "ファイル操作"),
    ),
    WorkspaceAreaSpec(
        "center_workspace",
        "中央作業ビュー",
        "Center Workspace",
        "primary",
        (
            "現在作業の主画面",
            "形状作図",
            "メッシュ確認",
            "ステージ入力",
            "解析条件入力",
            "結果図",
            "帳票確認",
            "詳細フォーム",
        ),
        ("補助説明だけの常設領域",),
    ),
    WorkspaceAreaSpec(
        "right_auxiliary",
        "右ペイン",
        "Right Auxiliary Pane",
        "auxiliary",
        (
            "現在作業の要約",
            "入力課題の場所",
            "入力課題の原因",
            "解消操作",
            "次の操作",
            "表示倍率",
            "変形倍率",
            "凡例の軽量調整",
        ),
        ("巨大フォーム", "長い表", "主結果図", "解析実行の主入口", "帳票生成の主入口"),
    ),
    WorkspaceAreaSpec(
        "bottom_primary_actions",
        "下部実行操作",
        "Bottom Run Actions",
        "execution_actions",
        ("解析実行", "結果リセット", "停止", "保存"),
        ("詳細入力フォーム", "監査", "内部診断"),
    ),
    WorkspaceAreaSpec(
        "menu_bar",
        "メニューバー",
        "Menu Bar",
        "global_commands",
        ("ファイル操作", "編集操作", "表示操作", "解析操作", "詳細設定", "ヘルプ"),
        ("作業カテゴリの常時切替", "左ツリーの代替", "中央フォームの複製"),
    ),
)


MENU_STRUCTURE: tuple[dict[str, Any], ...] = (
    {
        "id": "file",
        "label_ja": "ファイル",
        "label_en": "File",
        "purpose": "global_file_operations",
        "items": (
            "file.new_sample",
            "file.open_input",
            "file.save_input",
            "file.save_input_as",
            "file.import",
            "file.export",
            "file.recent",
            "file.exit",
        ),
    },
    {
        "id": "edit",
        "label_ja": "編集",
        "label_en": "Edit",
        "purpose": "current_workspace_editing",
        "items": (
            "edit.undo",
            "edit.redo",
            "edit.cut",
            "edit.copy",
            "edit.paste",
            "edit.delete",
            "edit.clear_selection",
            "edit.coordinate",
        ),
    },
    {
        "id": "view",
        "label_ja": "表示",
        "label_en": "View",
        "purpose": "view_state_control",
        "items": (
            "view.fit",
            "view.pan",
            "view.grid",
            "view.snap",
            "view.panels",
            "view.quality",
            "view.result_legend",
        ),
    },
    {
        "id": "analysis",
        "label_ja": "解析",
        "label_en": "Analysis",
        "purpose": "analysis_global_entry",
        "items": (
            "analysis.conditions",
            "analysis.check",
            "analysis.run",
            "analysis.stop",
            "analysis.reset_results",
            "analysis.convergence_log",
            "analysis.refresh_results",
        ),
    },
    {
        "id": "operations",
        "label_ja": "操作",
        "label_en": "Operations",
        "purpose": "command_search_and_secondary_entries",
        "items": ("command_catalog",),
    },
    {
        "id": "advanced_settings",
        "label_ja": "詳細設定",
        "label_en": "Advanced Settings",
        "purpose": "low_frequency_features",
        "items": (
            "settings.language",
            "settings.advanced_mode",
            "settings.maintenance_mode",
            "settings.audit",
            "settings.compatibility",
            "settings.diagnostics",
            "settings.yaml",
        ),
    },
)


PRIMARY_COMMAND_SURFACES: dict[str, str] = {
    "analysis.run": "bottom_primary_actions",
    "analysis.reset_results": "bottom_primary_actions",
    "analysis.stop": "bottom_primary_actions",
    "input.save": "bottom_primary_actions",
    "report.generate": "center_workspace",
    "file.new_sample": "menu_bar",
    "file.open_input": "menu_bar",
    "file.save_input_as": "menu_bar",
    "file.import": "menu_bar",
    "file.export": "menu_bar",
}


SECONDARY_COMMAND_SURFACES: dict[str, tuple[str, ...]] = {
    "analysis.run": ("menu_bar", "command_palette"),
    "analysis.reset_results": ("menu_bar",),
    "analysis.stop": ("menu_bar", "command_palette"),
    "input.save": ("menu_bar", "command_palette"),
    "report.generate": ("menu_bar", "command_palette"),
}


STANDARD_MODE_PANELS: tuple[str, ...] = (
    "workflow",
    "analysis",
    "geometry",
    "mesh",
    "materials",
    "boundary_conditions",
    "loads",
    "stages",
    "model_check",
    "results",
    "report",
)


DETAIL_MODE_PANELS: tuple[str, ...] = (
    "external",
    "solver",
    "yaml",
    "recovery",
    "audit",
    "internal_json",
    "compatibility_matrix",
    "verification_log",
)


VIEWPORT_REGRESSION_TARGETS: tuple[dict[str, int], ...] = (
    {"width": 2560, "height": 1440},
    {"width": 1600, "height": 900},
    {"width": 1366, "height": 768},
)


def workspace_responsibility_contract(*, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, Any]:
    """Return the stable responsibility boundary for visible GUI regions."""

    areas = []
    for area in WORKSPACE_AREAS:
        areas.append(
            {
                "id": area.id,
                "label": area.label_en if locale == "en" else area.label_ja,
                "role": area.role,
                "allowed": list(area.allowed),
                "disallowed": list(area.disallowed),
            }
        )
    return {
        "schema": WORKSPACE_CONTRACT_SCHEMA,
        "locale": locale,
        "areas": areas,
        "standard_mode_panels": list(STANDARD_MODE_PANELS),
        "detail_mode_panels": list(DETAIL_MODE_PANELS),
        "viewport_regression_targets": [dict(item) for item in VIEWPORT_REGRESSION_TARGETS],
        "features": {
            "navigation_boundary": True,
            "center_workspace_primary": True,
            "right_pane_auxiliary_only": True,
            "single_primary_command_surface": True,
            "standard_mode_frequency_filter": True,
            "issue_navigation_blocking": True,
            "viewport_regression": True,
        },
    }


def workspace_view_contract(*, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, Any]:
    """Return the work-view mapping used to keep tree, wizard, and heading in sync."""

    views = []
    for step in workflow_steps(locale=locale):
        views.append(
            {
                "id": step["id"],
                "label": step["label"],
                "panel": step["panel"],
                "workspace": "center_workspace",
                "heading_fields": ["work_name", "input_status", "next_action"],
                "main_region": True,
                "standard_visible": str(step["panel"]) in STANDARD_MODE_PANELS,
            }
        )
    return {
        "schema": "geofem.gui.workspace_view_contract.v1",
        "locale": locale,
        "views": views,
        "features": {
            "tree_wizard_heading_shared_ids": True,
            "details_return_to_center": True,
            "standard_mode_uses_common_views": True,
        },
    }


def menu_bar_contract(*, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, Any]:
    """Return the expected global menu grouping and command ownership."""

    menus = []
    for menu in MENU_STRUCTURE:
        menus.append(
            {
                "id": menu["id"],
                "label": menu["label_en"] if locale == "en" else menu["label_ja"],
                "purpose": menu["purpose"],
                "items": list(menu["items"]),
            }
        )
    return {
        "schema": "geofem.gui.menu_bar_contract.v1",
        "locale": locale,
        "menus": menus,
        "primary_command_surfaces": dict(PRIMARY_COMMAND_SURFACES),
        "secondary_command_surfaces": {key: list(value) for key, value in SECONDARY_COMMAND_SURFACES.items()},
        "features": {
            "file_menu_complete": True,
            "edit_menu_current_view_only": True,
            "view_menu_display_state_only": True,
            "analysis_menu_secondary_entry": True,
            "advanced_settings_after_operations": True,
        },
    }


def validate_workspace_contract(*, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, Any]:
    """Validate that the responsibility and menu contracts are internally consistent."""

    errors: list[str] = []
    contract = workspace_responsibility_contract(locale=locale)
    menu_contract = menu_bar_contract(locale=locale)
    area_ids = {str(area["id"]) for area in contract["areas"]}
    for required in ("left_tree", "center_workspace", "right_auxiliary", "bottom_primary_actions", "menu_bar"):
        if required not in area_ids:
            errors.append(f"missing workspace area: {required}")
    by_area = {str(area["id"]): area for area in contract["areas"]}
    if "解析実行" in by_area.get("left_tree", {}).get("allowed", []):
        errors.append("left tree allows analysis execution")
    if not {"巨大フォーム", "長い表", "主結果図"}.issubset(set(by_area.get("right_auxiliary", {}).get("disallowed", []))):
        errors.append("right auxiliary pane does not reject large primary content")
    menu_ids = [str(menu["id"]) for menu in menu_contract["menus"]]
    if len(menu_ids) != len(set(menu_ids)):
        errors.append("menu id is duplicated")
    if "operations" in menu_ids and "advanced_settings" in menu_ids:
        if menu_ids.index("advanced_settings") != menu_ids.index("operations") + 1:
            errors.append("advanced settings menu must follow operations")
    else:
        errors.append("operations or advanced settings menu is missing")
    primary_ids = list(PRIMARY_COMMAND_SURFACES)
    if len(primary_ids) != len(set(primary_ids)):
        errors.append("primary command id is duplicated")
    for command_id, surface in PRIMARY_COMMAND_SURFACES.items():
        if surface not in area_ids and surface != "command_palette":
            errors.append(f"{command_id} has unknown primary surface: {surface}")
    forbidden_standard = set(STANDARD_MODE_PANELS) & set(DETAIL_MODE_PANELS)
    if forbidden_standard:
        errors.append("panel appears in both standard and detail modes: " + ", ".join(sorted(forbidden_standard)))
    if len(VIEWPORT_REGRESSION_TARGETS) != 3:
        errors.append("viewport regression target count must be three")
    return {
        "schema": WORKSPACE_CONTRACT_VALIDATION_SCHEMA,
        "locale": locale,
        "passed": not errors,
        "errors": errors,
        "area_count": len(area_ids),
        "menu_count": len(menu_ids),
        "view_count": len(workflow_steps(locale=locale)),
        "viewport_count": len(VIEWPORT_REGRESSION_TARGETS),
    }


def write_workspace_contract(output_dir: str | Path, *, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, str]:
    """Write review artifacts for GUI responsibility regression."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "responsibility": workspace_responsibility_contract(locale=locale),
        "views": workspace_view_contract(locale=locale),
        "menus": menu_bar_contract(locale=locale),
        "validation": validate_workspace_contract(locale=locale),
    }
    json_path = out / "gui_workspace_contract.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"json": str(json_path)}


__all__ = [
    "DETAIL_MODE_PANELS",
    "PRIMARY_COMMAND_SURFACES",
    "SECONDARY_COMMAND_SURFACES",
    "STANDARD_MODE_PANELS",
    "VIEWPORT_REGRESSION_TARGETS",
    "menu_bar_contract",
    "validate_workspace_contract",
    "workspace_responsibility_contract",
    "workspace_view_contract",
    "write_workspace_contract",
]
