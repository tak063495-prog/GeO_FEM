"""Contextual GUI help policy for commercial-grade operation learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HelpPolicyResult:
    """Summary of widgets annotated with contextual help."""

    widget_count: int
    helped_count: int
    linked_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "widget_count": self.widget_count,
            "helped_count": self.helped_count,
            "linked_count": self.linked_count,
        }


PANEL_HELP: dict[str, dict[str, str]] = {
    "workspace": {
        "help_id": "geofem.help.workspace",
        "summary": "Main workspace, model view, command bar, and project actions.",
    },
    "workflow": {
        "help_id": "geofem.help.workflow",
        "summary": "Workflow guidance, missing inputs, operation log, and next action.",
    },
    "analysis": {
        "help_id": "geofem.help.analysis",
        "summary": "Analysis type, units, solver profile, and GeoFEAS/VGFlow substitute options.",
    },
    "geometry": {
        "help_id": "geofem.help.geometry",
        "summary": "Geometry, CAD import, regions, curves, repair checks, and snapping.",
    },
    "mesh": {
        "help_id": "geofem.help.mesh",
        "summary": "Mesh generation, element type, quality checks, and refinement controls.",
    },
    "materials": {
        "help_id": "geofem.help.materials",
        "summary": "Material models, constitutive parameters, and liquefaction settings.",
    },
    "external": {
        "help_id": "geofem.help.external",
        "summary": "Open exchange formats, CAD converters, and product-link diagnostics.",
    },
    "boundary_conditions": {
        "help_id": "geofem.help.boundary_conditions",
        "summary": "Supports, constraints, MPC, water level, and seepage boundary settings.",
    },
    "loads": {
        "help_id": "geofem.help.loads",
        "summary": "Loads, self weight, seismic actions, water pressure, and load cases.",
    },
    "stages": {
        "help_id": "geofem.help.stages",
        "summary": "Construction stages, activation, stress release, and staged reviews.",
    },
    "solver": {
        "help_id": "geofem.help.solver",
        "summary": "Solver controls, convergence settings, and nonlinear step controls.",
    },
    "model_check": {
        "help_id": "geofem.help.model_check",
        "summary": "Preflight checks, input errors, warnings, and automatic fixes.",
    },
    "results": {
        "help_id": "geofem.help.results",
        "summary": "Post views, result tables, contour/vector plots, and result exports.",
    },
    "report": {
        "help_id": "geofem.help.report",
        "summary": "Report generation, drawing layout, PDF/HTML output, and audit manifest.",
    },
    "yaml": {
        "help_id": "geofem.help.yaml",
        "summary": "Raw YAML input, fragment editors, schema-oriented editing, and sync.",
    },
}

DOCUMENTATION_LINKS: dict[str, dict[str, str]] = {
    "errors.input_cell": {
        "help_id": "geofem.help.errors.input_cell",
        "summary": "Input error cells, automatic repair, and how to jump back to the source table.",
    },
    "report.item": {
        "help_id": "geofem.help.report.item",
        "summary": "Report page items, layout fields, preview, and drawing/PDF output.",
    },
    "report.audit": {
        "help_id": "geofem.help.report.audit",
        "summary": "Report manifest, Post/report audit, and output traceability.",
    },
    "settings.input": {
        "help_id": "geofem.help.settings.input",
        "summary": "GUI settings, YAML synchronization, units, and input validation.",
    },
}


def help_policy_contract() -> dict[str, Any]:
    """Return the UI-independent contextual help contract."""

    return {
        "schema": "geofem.gui.help_policy.v1",
        "property_names": ["helpId", "helpUrl"],
        "tooltip_suffix": "Help:",
        "panels": {key: dict(value) for key, value in PANEL_HELP.items()},
        "document_links": {key: dict(value) for key, value in DOCUMENTATION_LINKS.items()},
    }


def help_catalog() -> dict[str, dict[str, str]]:
    """Return the panel help catalog."""

    return {key: _panel_help(key) for key in PANEL_HELP}


def documentation_link_catalog() -> dict[str, dict[str, str]]:
    """Return stable documentation links used by GUI errors and reports."""

    return {key: documentation_payload(key) for key in DOCUMENTATION_LINKS}


def documentation_payload(key: str) -> dict[str, str]:
    """Return a serializable help payload for a non-widget GUI item."""

    spec = dict(DOCUMENTATION_LINKS.get(key, DOCUMENTATION_LINKS["settings.input"]))
    spec["help_url"] = help_url_for(spec["help_id"])
    return spec


def help_url_for(help_id: str) -> str:
    """Return the user-guide URL for a help id."""

    return _help_url(help_id)


def apply_help_policy(root: Any, panel_key: str, qt: Mapping[str, Any]) -> HelpPolicyResult:
    """Attach short help text and stable help links to a widget tree."""

    widget_class = qt.get("QWidget")
    if root is None or widget_class is None:
        return HelpPolicyResult(0, 0, 0)
    widgets = [root]
    if hasattr(root, "findChildren"):
        widgets.extend(root.findChildren(widget_class))

    panel_help = _panel_help(panel_key)
    helped_count = 0
    linked_count = 0
    for widget in widgets:
        if not _is_help_target(widget, qt):
            continue
        help_id = str(widget.property("helpId") or panel_help["help_id"])
        help_url = str(widget.property("helpUrl") or _help_url(help_id))
        widget.setProperty("helpId", help_id)
        widget.setProperty("helpUrl", help_url)
        linked_count += 1
        if _apply_widget_tooltip(widget, panel_help, help_id):
            helped_count += 1
        if hasattr(widget, "accessibleDescription") and hasattr(widget, "setAccessibleDescription"):
            description = str(widget.accessibleDescription() or "").strip()
            if not description:
                widget.setAccessibleDescription(str(widget.toolTip() or "").strip())
    return HelpPolicyResult(len(widgets), helped_count, linked_count)


def _panel_help(panel_key: str) -> dict[str, str]:
    return dict(PANEL_HELP.get(panel_key, PANEL_HELP["workspace"]))


def _help_url(help_id: str) -> str:
    anchor = help_id.replace("geofem.help.", "").replace("_", "-").replace(".", "-")
    return f"docs/user_guide.md#{anchor}"


def _is_help_target(widget: Any, qt: Mapping[str, Any]) -> bool:
    target_classes = tuple(
        cls
        for cls in (
            qt.get("QPushButton"),
            qt.get("QLineEdit"),
            qt.get("QComboBox"),
            qt.get("QCheckBox"),
            qt.get("QTableWidget"),
            qt.get("QPlainTextEdit"),
            qt.get("QTreeWidget"),
            qt.get("QTabWidget"),
            qt.get("QGroupBox"),
        )
        if cls is not None
    )
    return bool(target_classes and isinstance(widget, target_classes))


def _apply_widget_tooltip(widget: Any, panel_help: Mapping[str, str], help_id: str) -> bool:
    if not hasattr(widget, "toolTip") or not hasattr(widget, "setToolTip"):
        return False
    current = str(widget.toolTip() or "").strip()
    if "Help:" in current:
        return False
    subject = _subject(widget)
    summary = str(panel_help["summary"])
    base = current or f"{subject}: {summary}"
    widget.setToolTip(f"{base}\nHelp: {help_id}")
    return True


def _subject(widget: Any) -> str:
    for accessor in ("text", "title", "objectName"):
        if not hasattr(widget, accessor):
            continue
        try:
            value = getattr(widget, accessor)()
        except TypeError:
            continue
        text = str(value or "").replace("&", "").strip()
        if text:
            return text
    return widget.__class__.__name__
