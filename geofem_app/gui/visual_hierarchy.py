"""Visual information hierarchy helpers for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class VisualHierarchyResult:
    """Summary of widgets marked with visual hierarchy roles."""

    widget_count: int
    layered_count: int

    def as_dict(self) -> dict[str, int]:
        return {"widget_count": self.widget_count, "layered_count": self.layered_count}


VISUAL_ROLE_SPECS: dict[str, dict[str, str | int]] = {
    "primary": {
        "priority": 1,
        "label": "Primary information",
        "description": "Model view, result view, report summary, or main data table.",
    },
    "warning": {
        "priority": 0,
        "label": "Warning and required attention",
        "description": "Errors, warnings, preflight checks, or input corrections.",
    },
    "auxiliary": {
        "priority": 2,
        "label": "Auxiliary information",
        "description": "Guidance, filters, selectors, and secondary controls.",
    },
    "detail": {
        "priority": 3,
        "label": "Detailed log",
        "description": "Logs, audit traces, histories, and verbose diagnostic output.",
    },
}


def visual_hierarchy_contract() -> dict[str, Any]:
    """Return the UI-independent information hierarchy contract."""

    return {
        "schema": "geofem.gui.visual_hierarchy.v1",
        "property_names": ["informationRole", "visualPriority"],
        "roles": {role: dict(spec) for role, spec in VISUAL_ROLE_SPECS.items()},
        "default_panel_roles": {
            "QTableWidget": "primary",
            "QTableView": "primary",
            "QPlainTextEdit": "detail",
            "QLabel": "auxiliary",
            "QGroupBox": "auxiliary",
            "QGraphicsView": "primary",
        },
    }


def mark_visual_role(widget: Any, role: str, description: str = "") -> bool:
    """Mark a widget with an information role and accessible description."""

    spec = VISUAL_ROLE_SPECS.get(role)
    if spec is None or widget is None:
        return False
    widget.setProperty("informationRole", role)
    widget.setProperty("visualPriority", int(spec["priority"]))
    note = description or str(spec["description"])
    if hasattr(widget, "accessibleDescription") and hasattr(widget, "setAccessibleDescription"):
        existing = str(widget.accessibleDescription() or "").strip()
        layer_note = f"Information layer: {spec['label']} - {note}"
        if layer_note not in existing:
            widget.setAccessibleDescription(f"{existing}\n{layer_note}".strip() if existing else layer_note)
    _refresh_widget_style(widget)
    return True


def apply_visual_hierarchy(root: Any, panel_key: str, qt: Mapping[str, Any]) -> VisualHierarchyResult:
    """Apply default visual hierarchy roles to a widget tree."""

    widget_class = qt.get("QWidget")
    if root is None or widget_class is None:
        return VisualHierarchyResult(0, 0)
    widgets = [root]
    if hasattr(root, "findChildren"):
        widgets.extend(root.findChildren(widget_class))
    layered_count = 0
    for widget in widgets:
        if widget.property("informationRole"):
            continue
        role = _default_role(widget, panel_key, qt)
        if role and mark_visual_role(widget, role):
            layered_count += 1
    return VisualHierarchyResult(len(widgets), layered_count)


def _default_role(widget: Any, panel_key: str, qt: Mapping[str, Any]) -> str:
    table_class = qt.get("QTableWidget")
    table_view_class = qt.get("QTableView")
    plain_text_class = qt.get("QPlainTextEdit")
    label_class = qt.get("QLabel")
    group_box_class = qt.get("QGroupBox")
    graphics_view_class = qt.get("QGraphicsView")
    table_classes = tuple(cls for cls in (table_class, table_view_class) if cls is not None)
    if table_classes and isinstance(widget, table_classes):
        if panel_key in {"model_check"} or _name_contains(widget, ("error", "warning", "check", "conflict")):
            return "warning"
        if _name_contains(widget, ("audit", "history", "log")):
            return "detail"
        return "primary"
    if plain_text_class is not None and isinstance(widget, plain_text_class):
        return "detail"
    if graphics_view_class is not None and isinstance(widget, graphics_view_class):
        return "primary"
    if label_class is not None and isinstance(widget, label_class):
        return "auxiliary"
    if group_box_class is not None and isinstance(widget, group_box_class):
        return "auxiliary"
    return ""


def _name_contains(widget: Any, needles: tuple[str, ...]) -> bool:
    name = str(widget.objectName() or "").lower() if hasattr(widget, "objectName") else ""
    return any(needle in name for needle in needles)


def _refresh_widget_style(widget: Any) -> None:
    if not hasattr(widget, "style"):
        return
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)
