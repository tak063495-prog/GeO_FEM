"""Model-check panel construction and issue rendering helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from geofem_app.gui.presentation_labels import friendly_input_reference


Issue = tuple[str, str, str, dict[str, Any]]


CHECK_TABLE_HEADERS = ("区分", "対象", "内容")
SEVERITY_BACKGROUNDS = {
    "ERROR": "#f8d7da",
    "WARN": "#fff3cd",
    "INFO": "#d1e7dd",
}


def model_check_panel_contract() -> dict[str, Any]:
    """Return a UI-independent description of the model-check panel."""

    return {
        "schema": "geofem.gui.model_check_panel.v1",
        "headers": list(CHECK_TABLE_HEADERS),
        "run_callback": "run_model_check_async",
        "summary_attribute": "check_summary",
        "table_attribute": "check_table",
        "severity_backgrounds": dict(SEVERITY_BACKGROUNDS),
    }


def build_model_check_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    """Build the model-check QWidget while keeping MainWindow layout-light."""

    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QLabel = qt["QLabel"]
    QTableWidget = qt["QTableWidget"]
    QHeaderView = qt["QHeaderView"]

    page = QWidget()
    layout = QVBoxLayout(page)
    owner.model_check_yaml_status = QLabel("YAML入力状態: 未確認")
    owner.model_check_yaml_status.setWordWrap(True)
    owner.check_summary = QLabel()
    owner.check_summary.setWordWrap(True)
    owner.model_check_workflow_table = QTableWidget(0, 4)
    owner.model_check_workflow_table.setHorizontalHeaderLabels(["工程", "状態", "不足/確認", "次の操作"])
    owner.model_check_workflow_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    owner.model_check_workflow_table.setMinimumHeight(180)
    owner.model_check_issue_table = QTableWidget(0, 3)
    owner.model_check_issue_table.setHorizontalHeaderLabels(list(CHECK_TABLE_HEADERS))
    owner.model_check_issue_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    owner.model_check_issue_table.setMinimumHeight(180)
    if hasattr(owner, "_model_check_issue_selection_changed"):
        owner.model_check_issue_table.itemSelectionChanged.connect(owner._model_check_issue_selection_changed)
    layout.addWidget(QLabel("YAML入力状態"))
    layout.addWidget(owner.model_check_yaml_status)
    layout.addWidget(QLabel("1解析から7ステージまでの確認状況"))
    layout.addWidget(owner.model_check_workflow_table)
    layout.addWidget(QLabel("モデルチェック結果"))
    layout.addWidget(owner.check_summary)
    layout.addWidget(owner.model_check_issue_table)
    layout.addStretch(1)
    return page


def summarize_model_check_issues(issues: Sequence[Sequence[Any]]) -> dict[str, int]:
    """Count model-check severities in a Qt-independent way."""

    errors = sum(1 for item in issues if item and item[0] == "ERROR")
    warnings = sum(1 for item in issues if item and item[0] == "WARN")
    infos = len(issues) - errors - warnings
    return {"ERROR": errors, "WARN": warnings, "INFO": infos, "TOTAL": len(issues)}


def _render_issue_table(
    table: Any,
    issues: list[tuple[str, str, str, dict[str, Any]]],
    qt: Mapping[str, Any],
    *,
    locale: str = "ja",
    show_internal: bool = False,
) -> None:
    QTableWidgetItem = qt["QTableWidgetItem"]
    Qt = qt["Qt"]
    QColor = qt["QColor"]

    table.setRowCount(0)
    for issue in issues:
        severity, target, detail = issue[:3]
        payload = dict(issue[3]) if len(issue) >= 4 and isinstance(issue[3], Mapping) else {}
        payload.setdefault("_raw_target", str(target))
        payload.setdefault("_raw_detail", str(detail))
        display_target = _localize_issue_text(target, locale=locale)
        if not show_internal:
            display_target = friendly_input_reference(display_target, locale=locale)
        row = table.rowCount()
        table.insertRow(row)
        for col, text in enumerate([severity, display_target, _localize_issue_text(detail, locale=locale)]):
            item = QTableWidgetItem(str(text))
            item.setData(Qt.ItemDataRole.UserRole, payload)
            item.setBackground(QColor(SEVERITY_BACKGROUNDS.get(str(severity), SEVERITY_BACKGROUNDS["INFO"])))
            table.setItem(row, col, item)


def apply_model_check_issues_view(owner: Any, issues: list[tuple[str, str, str, dict[str, Any]]], qt: Mapping[str, Any]) -> dict[str, int]:
    """Render model-check issues to the shared check table and summary label."""

    locale = str(getattr(owner, "gui_locale", "ja") or "ja")
    show_internal = bool(getattr(owner, "_show_internal_representation", lambda: False)())
    owner._last_model_check_issues = list(issues)
    _render_issue_table(owner.check_table, issues, qt, locale=locale, show_internal=show_internal)
    mirror = getattr(owner, "model_check_issue_table", None)
    if mirror is not None:
        _render_issue_table(mirror, issues, qt, locale=locale, show_internal=show_internal)
    counts = summarize_model_check_issues(issues)
    if hasattr(owner, "ui_message"):
        summary = owner.ui_message(
            "status.model_check.summary",
            errors=counts["ERROR"],
            warnings=counts["WARN"],
            infos=counts["INFO"],
        )
    else:
        summary = f"ERROR {counts['ERROR']} / WARN {counts['WARN']} / INFO {counts['INFO']}"
    owner.check_summary.setText(summary)
    if hasattr(owner, "refresh_model_check_overview"):
        owner.refresh_model_check_overview()
    owner.statusBar().showMessage(owner.check_summary.text())
    return counts


def _localize_issue_text(value: Any, *, locale: str) -> str:
    text = str(value)
    if not locale.startswith("en"):
        return text
    replacements = {
        "入力補助": "Input Assistance",
        "材料がありません。": "No materials defined.",
        "節点対象はマッピングで指定してください。": "Specify node targets as a mapping.",
        "拘束DOFが指定されていません。": "No constrained DOF is specified.",
    }
    if text in replacements:
        return replacements[text]
    if text.startswith("入力補助."):
        return "Input Assistance." + text.split(".", 1)[1]
    match = re.fullmatch(r"単位系 (.*?) / 補助項目 (\d+) 件", text)
    if match:
        return f"Unit system {match.group(1)} / guidance items {match.group(2)}"
    match = re.fullmatch(r"節点 (\d+) / 要素 (\d+) / インターフェース (\d+)", text)
    if match:
        return f"Nodes {match.group(1)} / Elements {match.group(2)} / Interfaces {match.group(3)}"
    match = re.fullmatch(r"(\d+) 件", text)
    if match:
        count = int(match.group(1))
        return f"{count} item{'s' if count != 1 else ''}"
    match = re.fullmatch(r"節点対象 (\d+) 件", text)
    if match:
        return f"Node targets {match.group(1)}"
    match = re.fullmatch(r"辺対象 (\d+) 件", text)
    if match:
        return f"Edge targets {match.group(1)}"
    match = re.fullmatch(r"拘束DOF (\d+)", text)
    if match:
        return f"Constrained DOF {match.group(1)}"
    match = re.fullmatch(r"未定義材料を参照する要素: (.*)", text)
    if match:
        return f"Elements referencing undefined materials: {match.group(1)}"
    match = re.fullmatch(r"材料未割当の対象: (.*)", text)
    if match:
        return f"Targets without material assignment: {match.group(1)}"
    return text
