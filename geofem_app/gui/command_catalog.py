"""Command hierarchy definitions for the desktop GUI."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from geofem_app.gui.i18n import DEFAULT_GUI_LOCALE, gui_message


ACTION_ROLE_KEYS = {
    "primary": "gui.action_role.primary",
    "confirm": "gui.action_role.confirm",
    "output": "gui.action_role.output",
    "detail": "gui.action_role.detail",
}
ACTION_ROLE_LABELS = {role: gui_message(key) for role, key in ACTION_ROLE_KEYS.items()}


@dataclass(frozen=True)
class GuiCommandSpec:
    id: str
    label_key: str
    role: str
    callback: str
    target_panel: str
    shortcut: str = ""
    toolbar: bool = False
    contexts: tuple[str, ...] = ()
    tooltip_key: str = ""


COMMAND_SPECS: tuple[GuiCommandSpec, ...] = (
    GuiCommandSpec("workflow.open", "gui.command.workflow.open.label", "primary", "show_workflow_panel", "workflow", "Ctrl+Shift+W", True, ("tree", "panel"), "gui.command.workflow.open.tooltip"),
    GuiCommandSpec("input.save", "gui.command.input.save.label", "primary", "save_input", "yaml", "Ctrl+S", True, ("tree", "panel"), "gui.command.input.save.tooltip"),
    GuiCommandSpec("analysis.run", "gui.command.analysis.run.label", "primary", "run_solver", "solver", "F5", True, ("tree", "view", "panel"), "gui.command.analysis.run.tooltip"),
    GuiCommandSpec("workflow.refresh", "gui.command.workflow.refresh.label", "confirm", "refresh_workflow_guidance", "workflow", "F7", True, ("tree", "panel"), "gui.command.workflow.refresh.tooltip"),
    GuiCommandSpec("model.check", "gui.command.model.check.label", "confirm", "run_model_check_async", "model_check", "F6", True, ("tree", "view", "panel"), "gui.command.model.check.tooltip"),
    GuiCommandSpec("analysis.stop", "gui.command.analysis.stop.label", "confirm", "stop_solver", "solver", "Shift+F5", True, ("tree", "panel"), "gui.command.analysis.stop.tooltip"),
    GuiCommandSpec("errors.jump", "gui.command.errors.jump.label", "confirm", "jump_to_selected_cell_error", "model_check", "Ctrl+E", False, ("panel",), "gui.command.errors.jump.tooltip"),
    GuiCommandSpec("errors.fix_all", "gui.command.errors.fix_all.label", "confirm", "fix_all_cell_errors", "model_check", "Ctrl+Shift+E", False, ("panel",), "gui.command.errors.fix_all.tooltip"),
    GuiCommandSpec("results.open", "gui.command.results.open.label", "output", "show_last_run_path", "results", "Ctrl+Shift+O", True, ("tree", "panel"), "gui.command.results.open.tooltip"),
    GuiCommandSpec("report.open", "gui.command.report.open.label", "output", "show_report_note", "report", "Ctrl+Shift+P", True, ("tree", "panel"), "gui.command.report.open.tooltip"),
    GuiCommandSpec("workspace.dashboard", "gui.command.workspace.dashboard.label", "output", "refresh_workspace_dashboard", "workflow", "Ctrl+Shift+D", False, ("tree", "panel"), "gui.command.workspace.dashboard.tooltip"),
    GuiCommandSpec("input.template.load", "gui.command.input.template.load.label", "detail", "load_project_template_input", "yaml", "", False, ("tree", "panel"), "gui.command.input.template.load.tooltip"),
    GuiCommandSpec("input.sync", "gui.command.input.sync.label", "detail", "sync_from_yaml", "yaml", "Ctrl+R", False, ("panel",), "gui.command.input.sync.tooltip"),
    GuiCommandSpec("customization.apply", "gui.command.customization.apply.label", "detail", "apply_default_customization_profile", "report", "", False, ("tree", "panel"), "gui.command.customization.apply.tooltip"),
    GuiCommandSpec("yaml.open", "gui.command.yaml.open.label", "detail", "show_yaml_panel", "yaml", "Ctrl+Alt+M", False, ("tree", "panel"), "gui.command.yaml.open.tooltip"),
    GuiCommandSpec("diagnostics.recovery", "gui.command.diagnostics.recovery.label", "detail", "refresh_recovery_candidates_table", "model_check", "Ctrl+Alt+R", False, ("panel",), "gui.command.diagnostics.recovery.tooltip"),
    GuiCommandSpec("diagnostics.audit", "gui.command.diagnostics.audit.label", "detail", "refresh_audit_log_table", "model_check", "Ctrl+Alt+A", False, ("panel",), "gui.command.diagnostics.audit.tooltip"),
)


def gui_command_catalog(*, locale: str = DEFAULT_GUI_LOCALE) -> list[dict[str, Any]]:
    """Return the stable commercial-grade GUI command catalog."""

    return [
        {
            "id": spec.id,
            "label": gui_message(spec.label_key, locale=locale),
            "label_key": spec.label_key,
            "role": spec.role,
            "role_label": gui_message(ACTION_ROLE_KEYS[spec.role], locale=locale),
            "role_label_key": ACTION_ROLE_KEYS[spec.role],
            "callback": spec.callback,
            "target_panel": spec.target_panel,
            "shortcut": spec.shortcut,
            "toolbar": spec.toolbar,
            "contexts": list(spec.contexts),
            "tooltip": gui_message(spec.tooltip_key, locale=locale) if spec.tooltip_key else "",
            "tooltip_key": spec.tooltip_key,
        }
        for spec in COMMAND_SPECS
    ]


def command_hierarchy(*, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, Any]:
    """Summarize commands by role, toolbar, context menu, and shortcuts."""

    commands = gui_command_catalog(locale=locale)
    roles = {role: [] for role in ACTION_ROLE_KEYS}
    contexts: dict[str, list[str]] = {"tree": [], "view": [], "panel": []}
    shortcuts: dict[str, str] = {}
    toolbar: list[str] = []
    for command in commands:
        roles[command["role"]].append(command)
        if command["toolbar"]:
            toolbar.append(command["id"])
        if command["shortcut"]:
            shortcuts[command["id"]] = command["shortcut"]
        for context in command["contexts"]:
            contexts.setdefault(context, []).append(command["id"])
    return {
        "schema": "geofem.gui.command_hierarchy.v1",
        "locale": locale,
        "role_labels": {role: gui_message(key, locale=locale) for role, key in ACTION_ROLE_KEYS.items()},
        "roles": roles,
        "toolbar": toolbar,
        "contexts": contexts,
        "shortcuts": shortcuts,
        "features": {
            "role_hierarchy": True,
            "toolbar": True,
            "menu": True,
            "context_menu": True,
            "shortcuts": True,
        },
    }


def validate_command_catalog(*, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, Any]:
    """Validate uniqueness and role coverage for the command catalog."""

    commands = gui_command_catalog(locale=locale)
    errors: list[str] = []
    ids = [command["id"] for command in commands]
    if len(ids) != len(set(ids)):
        errors.append("command id is duplicated")
    shortcuts = [command["shortcut"] for command in commands if command["shortcut"]]
    if len(shortcuts) != len(set(shortcuts)):
        errors.append("shortcut is duplicated")
    roles = {command["role"] for command in commands}
    missing_roles = set(ACTION_ROLE_KEYS) - roles
    if missing_roles:
        errors.append("missing roles: " + ", ".join(sorted(missing_roles)))
    missing_surface = [command["id"] for command in commands if not command["toolbar"] and not command["contexts"] and not command["shortcut"]]
    if missing_surface:
        errors.append("command has no operation surface: " + ", ".join(missing_surface))
    untranslated = [
        command["id"]
        for command in commands
        if command["label"] == command.get("label_key") or (command.get("tooltip_key") and command["tooltip"] == command.get("tooltip_key"))
    ]
    if untranslated:
        errors.append("untranslated command keys: " + ", ".join(untranslated))
    return {
        "schema": "geofem.gui.command_catalog_validation.v1",
        "locale": locale,
        "passed": not errors,
        "errors": errors,
        "command_count": len(commands),
        "role_count": len(roles),
        "shortcut_count": len(shortcuts),
    }


def write_command_catalog(output_dir: str | Path, *, locale: str = DEFAULT_GUI_LOCALE) -> dict[str, str]:
    """Write command catalog artifacts for GUI review and regression."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hierarchy = command_hierarchy(locale=locale)
    json_path = out / "gui_command_hierarchy.json"
    csv_path = out / "gui_command_catalog.csv"
    html_path = out / "gui_command_catalog.html"
    json_path.write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
    commands = gui_command_catalog(locale=locale)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "role_label", "target_panel", "shortcut", "toolbar", "contexts", "tooltip"])
        writer.writeheader()
        for command in commands:
            writer.writerow({key: _csv_value(command.get(key, "")) for key in writer.fieldnames})
    html_path.write_text(_html(commands, locale=locale), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _html(commands: list[dict[str, Any]], *, locale: str = DEFAULT_GUI_LOCALE) -> str:
    rows = []
    for command in commands:
        rows.append(
            "<tr>"
            f"<td>{escape(str(command['id']))}</td>"
            f"<td>{escape(str(command['label']))}</td>"
            f"<td>{escape(str(command['role_label']))}</td>"
            f"<td>{escape(str(command['target_panel']))}</td>"
            f"<td>{escape(str(command['shortcut']))}</td>"
            f"<td>{escape(', '.join(command['contexts']))}</td>"
            f"<td>{escape(str(command['tooltip']))}</td>"
            "</tr>"
        )
    return (
        f"<!doctype html><html lang=\"{escape(locale)}\"><meta charset=\"utf-8\">"
        "<title>GeoFEM GUI command catalog</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}th{background:#f4f4f4}</style>"
        f"<h1>{escape(gui_message('command.report.title', locale=locale))}</h1>"
        "<table><thead><tr>"
        f"<th>{escape(gui_message('command.table.id', locale=locale))}</th>"
        f"<th>{escape(gui_message('command.table.label', locale=locale))}</th>"
        f"<th>{escape(gui_message('command.table.role', locale=locale))}</th>"
        f"<th>{escape(gui_message('command.table.target', locale=locale))}</th>"
        f"<th>{escape(gui_message('command.table.shortcut', locale=locale))}</th>"
        f"<th>{escape(gui_message('command.table.context', locale=locale))}</th>"
        f"<th>{escape(gui_message('command.table.tooltip', locale=locale))}</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></html>"
    )


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)
