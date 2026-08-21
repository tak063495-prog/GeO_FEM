"""Accessibility helpers for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AccessibilityPolicyResult:
    """Summary of widgets touched by the accessibility policy."""

    widget_count: int
    named_count: int
    focusable_count: int
    described_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "widget_count": self.widget_count,
            "named_count": self.named_count,
            "focusable_count": self.focusable_count,
            "described_count": self.described_count,
        }


def accessibility_policy_contract() -> dict[str, Any]:
    """Return a UI-independent description of the GUI accessibility policy."""

    return {
        "schema": "geofem.gui.accessibility_policy.v1",
        "focus_policy": "StrongFocus",
        "named_widgets": [
            "QPushButton",
            "QLineEdit",
            "QComboBox",
            "QCheckBox",
            "QTableWidget",
            "QTableView",
            "QPlainTextEdit",
            "QTreeWidget",
            "QTabWidget",
        ],
        "severity_channels": ["accessibleDescription", "fontWeight", "background", "border"],
    }


def apply_accessibility_policy(root: Any, qt: Mapping[str, Any]) -> AccessibilityPolicyResult:
    """Apply accessible names, descriptions, and keyboard focus to a widget tree."""

    widget_class = qt.get("QWidget")
    if root is None or widget_class is None:
        return AccessibilityPolicyResult(0, 0, 0, 0)
    widgets = [root]
    if hasattr(root, "findChildren"):
        widgets.extend(root.findChildren(widget_class))

    named_count = 0
    focusable_count = 0
    described_count = 0
    focus_classes = tuple(
        cls
        for cls in (
            qt.get("QPushButton"),
            qt.get("QLineEdit"),
            qt.get("QComboBox"),
            qt.get("QCheckBox"),
            qt.get("QTableWidget"),
            qt.get("QTableView"),
            qt.get("QPlainTextEdit"),
            qt.get("QTreeWidget"),
            qt.get("QTabWidget"),
        )
        if cls is not None
    )
    for widget in widgets:
        label = _widget_label(widget)
        if label and not str(widget.accessibleName() or "").strip():
            widget.setAccessibleName(label)
            named_count += 1
        description = _widget_description(widget)
        if description and not str(widget.accessibleDescription() or "").strip():
            widget.setAccessibleDescription(description)
            described_count += 1
        if focus_classes and isinstance(widget, focus_classes):
            _set_strong_focus(widget, qt)
            focusable_count += 1
    return AccessibilityPolicyResult(len(widgets), named_count, focusable_count, described_count)


def _widget_label(widget: Any) -> str:
    for accessor in ("text", "title", "windowTitle", "toolTip", "objectName"):
        if not hasattr(widget, accessor):
            continue
        try:
            value = getattr(widget, accessor)()
        except TypeError:
            continue
        text = _clean_text(value)
        if text:
            return text
    return widget.__class__.__name__


def _widget_description(widget: Any) -> str:
    tooltip = _clean_text(widget.toolTip()) if hasattr(widget, "toolTip") else ""
    severity = _clean_text(widget.property("severity")) if hasattr(widget, "property") else ""
    if severity and tooltip:
        return f"{tooltip} / severity: {severity}"
    if severity:
        return f"severity: {severity}"
    return tooltip


def _set_strong_focus(widget: Any, qt: Mapping[str, Any]) -> None:
    qt_core = qt.get("Qt")
    if qt_core is None:
        return
    focus_policy = getattr(qt_core, "FocusPolicy", None)
    strong_focus = getattr(focus_policy, "StrongFocus", None) if focus_policy is not None else None
    if strong_focus is not None and hasattr(widget, "setFocusPolicy"):
        widget.setFocusPolicy(strong_focus)


def _clean_text(value: Any) -> str:
    return str(value or "").replace("&", "").strip()
