"""Commercial-grade GUI design tokens and stylesheet helpers."""

from __future__ import annotations

import os
from typing import Any


DESIGN_TOKENS: dict[str, Any] = {
    "schema": "geofem.gui.design_tokens.v1",
    "colors": {
        "window": "#f6f7f8",
        "surface": "#ffffff",
        "surface_subtle": "#eef2f5",
        "border": "#c9d1d8",
        "text": "#1f2933",
        "muted_text": "#5f6c78",
        "primary": "#22577a",
        "primary_hover": "#17425e",
        "focus": "#2d8fdd",
        "ok": "#1b7f4a",
        "warning": "#ad6b00",
        "error": "#b42318",
        "error_surface": "#fbe7e5",
        "warning_surface": "#fff4d6",
        "ok_surface": "#e4f4ea",
        "selection": "#d7ebf8",
    },
    "metrics": {
        "font_pt": 11,
        "compact_row_height": 26,
        "button_min_height": 32,
        "workflow_button_min_height": 36,
        "field_min_height": 30,
        "panel_padding_px": 8,
        "group_radius_px": 4,
        "table_grid_px": 1,
    },
    "font_family": "'Yu Gothic UI', 'Meiryo', 'Noto Sans CJK JP', 'Segoe UI', sans-serif",
    "states": {
        "error": {"color": "#b42318", "background": "#fbe7e5"},
        "warning": {"color": "#ad6b00", "background": "#fff4d6"},
        "ok": {"color": "#1b7f4a", "background": "#e4f4ea"},
        "info": {"color": "#22577a", "background": "#e8f2f8"},
    },
}


def resolve_gui_font_pt(base_font_pt: int | None = None, env: dict[str, str] | None = None) -> int:
    """Resolve the GUI font point size with bounded accessibility overrides."""

    source = os.environ if env is None else env
    base = int(base_font_pt or DESIGN_TOKENS["metrics"]["font_pt"])
    raw_pt = str(source.get("GEOFEM_GUI_FONT_PT", "") or "").strip()
    if raw_pt:
        try:
            return max(9, min(18, int(round(float(raw_pt)))))
        except ValueError:
            return base
    raw_scale = str(source.get("GEOFEM_GUI_FONT_SCALE", "") or "").strip()
    if not raw_scale:
        return base
    try:
        scale = max(1.0, min(1.6, float(raw_scale)))
    except ValueError:
        return base
    return max(9, min(18, int(round(base * scale))))


def commercial_design_tokens() -> dict[str, Any]:
    """Return a copy of the shared GUI design tokens."""

    metrics = dict(DESIGN_TOKENS["metrics"])
    metrics["font_pt"] = resolve_gui_font_pt(metrics["font_pt"])
    return {
        "schema": DESIGN_TOKENS["schema"],
        "colors": dict(DESIGN_TOKENS["colors"]),
        "metrics": metrics,
        "font_family": str(DESIGN_TOKENS["font_family"]),
        "states": {key: dict(value) for key, value in DESIGN_TOKENS["states"].items()},
    }


def commercial_gui_stylesheet() -> str:
    """Build the application-wide stylesheet from shared tokens."""

    tokens = commercial_design_tokens()
    colors = tokens["colors"]
    metrics = tokens["metrics"]
    font_family = tokens["font_family"]
    return f"""
QMainWindow, QWidget {{
    background: {colors["window"]};
    color: {colors["text"]};
    font-family: {font_family};
    font-size: {metrics["font_pt"]}pt;
}}
QToolBar {{
    background: {colors["surface_subtle"]};
    border: 0;
    border-bottom: 1px solid {colors["border"]};
    spacing: 4px;
    padding: 2px 6px;
}}
QToolBar::separator {{
    width: 1px;
    background: {colors["border"]};
    margin: 4px 6px;
}}
QToolButton {{
    min-height: 24px;
    padding: 3px 8px;
    border: 1px solid transparent;
    border-radius: 3px;
    background: transparent;
}}
QToolButton:hover {{
    border-color: {colors["border"]};
    background: {colors["surface"]};
}}
QLabel[role="toolbarLabel"] {{
    color: {colors["muted_text"]};
    font-weight: 600;
    padding: 0 8px 0 2px;
    background: transparent;
}}
QGroupBox {{
    background: {colors["surface"]};
    border: 1px solid {colors["border"]};
    border-radius: {metrics["group_radius_px"]}px;
    margin-top: 10px;
    padding: {metrics["panel_padding_px"]}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {colors["muted_text"]};
}}
QPushButton {{
    min-height: {metrics["button_min_height"]}px;
    padding: 5px 10px;
    border: 1px solid {colors["border"]};
    border-radius: 4px;
    background: {colors["surface"]};
}}
QPushButton[role="workflow"] {{
    min-height: {metrics["workflow_button_min_height"]}px;
    padding: 7px 12px;
    text-align: left;
    font-weight: 500;
}}
QWidget[role="workflowRibbon"] {{
    background: {colors["surface"]};
    border-bottom: 1px solid {colors["border"]};
}}
QPushButton[role="workflow_step"] {{
    min-height: 20px;
    max-height: 24px;
    padding: 1px 6px;
    font-weight: 500;
}}
QPushButton[role="workflow_ribbon_button"] {{
    min-height: 24px;
    max-height: 28px;
    padding: 2px 8px;
    font-weight: 500;
}}
QPushButton[role="workspace_focus_button"] {{
    min-height: 24px;
    max-height: 28px;
    padding: 2px 8px;
    font-weight: 600;
    border-color: {colors["primary"]};
}}
QPushButton[role="workspace_focus_button"]:checked {{
    color: {colors["surface"]};
    background: {colors["primary"]};
}}
QPushButton[role="panelAction"] {{
    min-height: {metrics["button_min_height"]}px;
    padding: 5px 9px;
    text-align: left;
}}
QPushButton:hover {{
    border-color: {colors["primary"]};
    background: {colors["surface_subtle"]};
}}
QPushButton:pressed {{
    background: {colors["selection"]};
}}
QPushButton:focus, QToolButton:focus {{
    border: 2px solid {colors["focus"]};
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {colors["muted_text"]};
    background: {colors["surface_subtle"]};
    border-color: {colors["border"]};
}}
QLineEdit, QComboBox, QPlainTextEdit {{
    min-height: {metrics["field_min_height"]}px;
    border: 1px solid {colors["border"]};
    border-radius: 3px;
    background: {colors["surface"]};
    selection-background-color: {colors["selection"]};
}}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTableWidget:focus, QTableView:focus {{
    border: 2px solid {colors["focus"]};
}}
QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{
    color: {colors["muted_text"]};
    background: {colors["surface_subtle"]};
}}
QCheckBox:focus {{
    color: {colors["primary"]};
}}
QCheckBox::indicator:focus {{
    border: 2px solid {colors["focus"]};
}}
QTableWidget, QTableView {{
    background: {colors["surface"]};
    gridline-color: {colors["border"]};
    alternate-background-color: {colors["surface_subtle"]};
    selection-background-color: {colors["selection"]};
}}
QTableWidget::item, QTableView::item {{
    min-height: {metrics["compact_row_height"]}px;
    padding: 3px 5px;
}}
QWidget[informationRole="primary"], QGraphicsView[informationRole="primary"], QPlainTextEdit[informationRole="primary"], QTableWidget[informationRole="primary"], QTableView[informationRole="primary"] {{
    border-left: 4px solid {colors["primary"]};
    background: {colors["surface"]};
}}
QWidget[informationRole="warning"], QLabel[informationRole="warning"], QTableWidget[informationRole="warning"], QTableView[informationRole="warning"] {{
    border-left: 4px solid {colors["warning"]};
    background: {colors["warning_surface"]};
}}
QWidget[informationRole="auxiliary"], QLabel[informationRole="auxiliary"], QGroupBox[informationRole="auxiliary"] {{
    border-left: 3px solid {colors["border"]};
    background: {colors["surface_subtle"]};
}}
QGroupBox[informationRole="primary"] {{
    border-left: 4px solid {colors["primary"]};
}}
QGroupBox[informationRole="auxiliary"] {{
    background: {colors["surface"]};
    border-color: {colors["surface_subtle"]};
}}
QWidget[informationRole="detail"], QPlainTextEdit[informationRole="detail"], QTableWidget[informationRole="detail"], QTableView[informationRole="detail"] {{
    border-left: 3px solid {colors["muted_text"]};
    background: {colors["surface_subtle"]};
    color: {colors["muted_text"]};
}}
QScrollArea {{
    border: 0;
    background: {colors["window"]};
}}
QHeaderView::section {{
    background: {colors["surface_subtle"]};
    border: 0;
    border-right: {metrics["table_grid_px"]}px solid {colors["border"]};
    border-bottom: {metrics["table_grid_px"]}px solid {colors["border"]};
    padding: 4px 6px;
    font-weight: 600;
}}
QTabWidget::pane {{
    border: 1px solid {colors["border"]};
    background: {colors["surface"]};
}}
QTabBar::tab {{
    padding: 6px 10px;
    border: 1px solid {colors["border"]};
    border-bottom: 0;
    background: {colors["surface_subtle"]};
}}
QTabBar::tab:selected {{
    background: {colors["surface"]};
    color: {colors["primary"]};
    font-weight: 600;
}}
QStatusBar {{
    background: {colors["surface_subtle"]};
    border-top: 1px solid {colors["border"]};
    color: {colors["muted_text"]};
}}
QLabel#notification_banner {{
    min-height: 28px;
    padding: 6px 10px;
    border: 1px solid {colors["border"]};
    border-left: 4px solid {colors["primary"]};
    border-radius: 4px;
    background: {colors["surface_subtle"]};
    color: {colors["text"]};
    font-weight: 600;
}}
QLabel#notification_banner[severity="info"] {{
    border-left-color: {colors["primary"]};
}}
QLabel[severity="error"], QStatusBar[severity="error"] {{
    color: {colors["error"]};
    background: {colors["error_surface"]};
    border-left: 4px solid {colors["error"]};
    padding-left: 6px;
    font-weight: 600;
}}
QLabel[severity="warning"], QStatusBar[severity="warning"] {{
    color: {colors["warning"]};
    background: {colors["warning_surface"]};
    border-left: 4px solid {colors["warning"]};
    padding-left: 6px;
    font-weight: 600;
}}
QLabel[severity="ok"], QStatusBar[severity="ok"] {{
    color: {colors["ok"]};
    background: {colors["ok_surface"]};
    border-left: 4px solid {colors["ok"]};
    padding-left: 6px;
    font-weight: 600;
}}
"""


def apply_commercial_gui_style(app: Any) -> None:
    """Apply the shared commercial GUI stylesheet to a Qt application."""

    if app is None:
        return
    stylesheet = commercial_gui_stylesheet()
    current = str(app.styleSheet() or "")
    if "geofem.gui.design_tokens.v1" not in current:
        app.setStyleSheet(f"/* geofem.gui.design_tokens.v1 */\n{stylesheet}")


__all__ = ["commercial_design_tokens", "commercial_gui_stylesheet", "apply_commercial_gui_style", "resolve_gui_font_pt"]
