"""Shared button icon policy for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ButtonIconSpec:
    """Qt standard icon mapping for a primary GUI command."""

    icon_role: str
    standard_pixmap: str
    tooltip: str


BUTTON_ICON_SPECS: dict[str, ButtonIconSpec] = {
    "view.reset": ButtonIconSpec("view.reset", "SP_BrowserReload", "モデル全体を表示します。"),
    "model.check": ButtonIconSpec("model.check", "SP_MessageBoxInformation", "入力とモデルを事前チェックします。"),
    "draw.line": ButtonIconSpec("draw.line", "SP_FileDialogDetailedView", "線分を作図します。"),
    "draw.region": ButtonIconSpec("draw.region", "SP_DirIcon", "閉領域を作図します。"),
    "draw.finish_region": ButtonIconSpec("draw.finish_region", "SP_DialogApplyButton", "作図中の領域を閉じます。"),
    "selection.rectangle": ButtonIconSpec("selection.rectangle", "SP_FileDialogDetailedView", "矩形範囲で選択します。"),
    "selection.lasso": ButtonIconSpec("selection.lasso", "SP_FileDialogContentsView", "投げ縄範囲で選択します。"),
    "selection.filter": ButtonIconSpec("selection.filter", "SP_ComputerIcon", "条件で対象を選択します。"),
    "selection.clear": ButtonIconSpec("selection.clear", "SP_DialogResetButton", "現在の選択を解除します。"),
    "selection.invert": ButtonIconSpec("selection.invert", "SP_BrowserReload", "選択状態を反転します。"),
    "selection.history": ButtonIconSpec("selection.history", "SP_FileIcon", "選択履歴を記録します。"),
    "selection.named": ButtonIconSpec("selection.named", "SP_DialogSaveButton", "選択セットに名前を付けて保存します。"),
    "selection.compare": ButtonIconSpec("selection.compare", "SP_FileDialogDetailedView", "保存済み選択セットを比較します。"),
    "selection.help": ButtonIconSpec("selection.help", "SP_DialogHelpButton", "選択モードの説明を表示します。"),
    "selection.save_set": ButtonIconSpec("selection.save_set", "SP_DialogSaveButton", "現在の選択をセットへ登録します。"),
    "selection.expression.build": ButtonIconSpec("selection.expression.build", "SP_FileDialogDetailedView", "条件式を作成します。"),
    "selection.expression.run": ButtonIconSpec("selection.expression.run", "SP_DialogApplyButton", "条件式で選択します。"),
    "analysis.run": ButtonIconSpec("analysis.run", "SP_DialogApplyButton", "解析を実行します。"),
    "analysis.reset_results": ButtonIconSpec("analysis.reset_results", "SP_DialogResetButton", "解析結果の表示状態をリセットし、入力編集に戻ります。"),
    "analysis.stop": ButtonIconSpec("analysis.stop", "SP_DialogCancelButton", "実行中の解析を停止します。"),
    "yaml.sync": ButtonIconSpec("yaml.sync", "SP_BrowserReload", "YAML編集内容を画面へ反映します。"),
    "undo": ButtonIconSpec("undo", "SP_ArrowBack", "直前の編集を元に戻します。"),
    "redo": ButtonIconSpec("redo", "SP_ArrowForward", "元に戻した編集をやり直します。"),
    "project.save": ButtonIconSpec("project.save", "SP_DialogSaveButton", "入力データを保存します。"),
    "edit.undo_point": ButtonIconSpec("edit.undo_point", "SP_ArrowLeft", "手動の戻し点を作成します。"),
    "project.lock": ButtonIconSpec("project.lock", "SP_DialogApplyButton", "プロジェクトをロックします。"),
    "project.unlock": ButtonIconSpec("project.unlock", "SP_DialogCancelButton", "プロジェクトロックを解除します。"),
    "project.force_unlock": ButtonIconSpec("project.force_unlock", "SP_MessageBoxWarning", "強制ロック解除を確認します。"),
    "project.handoff": ButtonIconSpec("project.handoff", "SP_ArrowRight", "ロックを引き継ぎます。"),
    "error.jump": ButtonIconSpec("error.jump", "SP_MessageBoxWarning", "選択中の入力エラーセルへ移動します。"),
    "error.fix_all": ButtonIconSpec("error.fix_all", "SP_DialogApplyButton", "修正可能な入力エラーを一括修正します。"),
    "recovery.open": ButtonIconSpec("recovery.open", "SP_DirOpenIcon", "復旧候補を更新して表示します。"),
    "audit.open": ButtonIconSpec("audit.open", "SP_FileDialogDetailedView", "監査ログを表示します。"),
    "panel.detail": ButtonIconSpec("panel.detail", "SP_FileDialogDetailedView", "選択中の作業の詳細画面を開きます。"),
    "panel.return": ButtonIconSpec("panel.return", "SP_ArrowLeft", "直前の標準表示またはCAD風作図パレットへ戻ります。"),
    "stage.add": ButtonIconSpec("stage.add", "SP_FileDialogNewFolder", "ステージを追加します。"),
    "stage.manage": ButtonIconSpec("stage.manage", "SP_FileDialogListView", "選択中のステージを整理します。"),
    "stage.change.add": ButtonIconSpec("stage.change.add", "SP_FileDialogNewFolder", "選択中のステージに変更を追加します。"),
    "stage.change.delete": ButtonIconSpec("stage.change.delete", "SP_TrashIcon", "選択したステージ変更を削除します。"),
    "stage.change.apply": ButtonIconSpec("stage.change.apply", "SP_DialogApplyButton", "ステージ変更を入力データに反映します。"),
    "result.visual": ButtonIconSpec("result.visual", "SP_DesktopIcon", "代表的な解析結果図を開きます。"),
    "result.report": ButtonIconSpec("result.report", "SP_FileIcon", "帳票画面を開きます。"),
    "result.folder": ButtonIconSpec("result.folder", "SP_DirOpenIcon", "現在参照中の解析結果を確認します。"),
    "result.srm": ButtonIconSpec("result.srm", "SP_MessageBoxInformation", "SRMの安全率と試行結果を確認します。"),
    "result.export": ButtonIconSpec("result.export", "SP_DialogSaveButton", "表示中の結果を出力します。"),
}


PANEL_DEFAULT_PIXMAPS: dict[str, str] = {
    "workflow": "SP_FileDialogDetailedView",
    "analysis": "SP_ComputerIcon",
    "geometry": "SP_DirIcon",
    "mesh": "SP_FileDialogDetailedView",
    "materials": "SP_FileIcon",
    "external": "SP_DirOpenIcon",
    "boundary_conditions": "SP_DialogApplyButton",
    "loads": "SP_ArrowRight",
    "stages": "SP_FileDialogDetailedView",
    "solver": "SP_ComputerIcon",
    "model_check": "SP_MessageBoxInformation",
    "results": "SP_DirOpenIcon",
    "report": "SP_DialogSaveButton",
    "yaml": "SP_FileIcon",
}


def button_icon_catalog() -> dict[str, dict[str, str]]:
    """Return a serializable catalog for tests and diagnostics."""

    return {
        key: {
            "icon_role": spec.icon_role,
            "standard_pixmap": spec.standard_pixmap,
            "tooltip": spec.tooltip,
        }
        for key, spec in BUTTON_ICON_SPECS.items()
    }


def panel_button_icon_catalog() -> dict[str, str]:
    """Return the default icon policy for panel-local action buttons."""

    return dict(PANEL_DEFAULT_PIXMAPS)


def apply_button_icon(button: Any, icon_role: str, style: Any, qstyle: Any) -> bool:
    """Apply a Qt standard icon to a QPushButton-like object."""

    spec = BUTTON_ICON_SPECS.get(icon_role)
    if spec is None:
        return False
    standard_pixmap = getattr(qstyle.StandardPixmap, spec.standard_pixmap, None)
    if standard_pixmap is None:
        return False
    button.setIcon(style.standardIcon(standard_pixmap))
    button.setToolTip(spec.tooltip)
    button.setProperty("iconRole", spec.icon_role)
    return True


def apply_panel_button_icon(button: Any, panel_key: str, style: Any, qstyle: Any) -> bool:
    """Apply the shared panel-action presentation to a QPushButton-like object."""

    standard_pixmap_name = PANEL_DEFAULT_PIXMAPS.get(panel_key, "SP_FileDialogDetailedView")
    standard_pixmap = getattr(qstyle.StandardPixmap, standard_pixmap_name, None)
    if standard_pixmap is None:
        return False
    if button.icon().isNull():
        button.setIcon(style.standardIcon(standard_pixmap))
    if not button.property("iconRole"):
        button.setProperty("iconRole", f"panel.{panel_key}")
    if not button.property("role"):
        button.setProperty("role", "panelAction")
    if button.minimumHeight() < 32:
        button.setMinimumHeight(32)
    if not button.toolTip():
        text = str(button.text() or "").replace("&", "").strip()
        button.setToolTip(text or panel_key)
    return True
