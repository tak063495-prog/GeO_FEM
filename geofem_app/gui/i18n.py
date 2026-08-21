"""Localized GUI wording for shared desktop surfaces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GUI_LOCALE = "ja"
SUPPORTED_GUI_LOCALES = ("ja", "en")


GUI_MESSAGES: dict[str, dict[str, str]] = {
    "ja": {
        "gui.language.ja": "日本語",
        "gui.language.en": "英語",
        "gui.menu.operations": "操作",
        "gui.menu.advanced_settings": "詳細設定",
        "gui.menu.language": "言語",
        "gui.language.switch_to_ja": "日本語に切り替え",
        "gui.language.switch_to_en": "英語に切り替え",
        "gui.toolbar.main": "主要操作",
        "gui.group.primary": "実行",
        "gui.group.confirm": "確認",
        "gui.group.output": "出力",
        "gui.group.detail": "詳細設定",
        "gui.group.project": "プロジェクト",
        "gui.label.undo_granularity": "戻し粒度",
        "gui.action_role.primary": "主要操作",
        "gui.action_role.confirm": "確認",
        "gui.action_role.output": "出力",
        "gui.action_role.detail": "詳細設定",
        "gui.command.palette.label": "コマンドパレット...",
        "gui.command.palette.tooltip": "必要なコマンドをポップアップで選択して実行します。",
        "gui.command.workflow.open.label": "ワークフロー",
        "gui.command.workflow.open.tooltip": "全体の作業順と不足箇所を表示します。",
        "gui.command.input.save.label": "保存",
        "gui.command.input.save.tooltip": "現在の入力を保存します。",
        "gui.command.analysis.run.label": "解析実行",
        "gui.command.analysis.run.tooltip": "モデルチェック後に解析を実行します。",
        "gui.command.workflow.refresh.label": "ワークフロー診断更新",
        "gui.command.workflow.refresh.tooltip": "必須入力充足率と次操作を更新します。",
        "gui.command.model.check.label": "モデルチェック",
        "gui.command.model.check.tooltip": "解析前の入力エラーと警告を確認します。",
        "gui.command.analysis.stop.label": "停止",
        "gui.command.analysis.stop.tooltip": "実行中の解析を停止します。",
        "gui.command.errors.jump.label": "エラーセルへ",
        "gui.command.errors.jump.tooltip": "選択中の入力エラー位置へ移動します。",
        "gui.command.errors.fix_all.label": "エラー一括修正",
        "gui.command.errors.fix_all.tooltip": "自動修正可能な入力エラーをまとめて修正します。",
        "gui.command.results.open.label": "結果確認",
        "gui.command.results.open.tooltip": "直近解析の結果を開きます。",
        "gui.command.report.open.label": "計算書",
        "gui.command.report.open.tooltip": "帳票出力とmanifestを確認します。",
        "gui.command.workspace.dashboard.label": "ワークスペース概要",
        "gui.command.workspace.dashboard.tooltip": "プロジェクト、最近の解析、成果物、容量、アーカイブ候補をまとめて出力します。",
        "gui.command.input.template.load.label": "入力テンプレート読込",
        "gui.command.input.template.load.tooltip": "形状、材料、境界、荷重、ステージを含む入力テンプレートを読込みます。",
        "gui.command.input.sync.label": "YAML反映",
        "gui.command.input.sync.tooltip": "YAML編集内容をフォームへ反映します。",
        "gui.command.customization.apply.label": "組織プロファイル適用",
        "gui.command.customization.apply.tooltip": "組織ロゴ、帳票テンプレート、単位系、色、既定値を現在の入力へ適用します。",
        "gui.command.yaml.open.label": "YAMLを開く",
        "gui.command.yaml.open.tooltip": "詳細な入力YAMLを開きます。",
        "gui.command.diagnostics.recovery.label": "復旧候補",
        "gui.command.diagnostics.recovery.tooltip": "自動保存から復旧候補を確認します。",
        "gui.command.diagnostics.audit.label": "監査ログ",
        "gui.command.diagnostics.audit.tooltip": "操作監査ログを確認します。",
        "workflow.step.analysis.label": "解析条件",
        "workflow.step.analysis.next": "解析種別、次元、単位系を確認します。",
        "workflow.step.analysis.detail": "解析種別と2D指定を確認します。",
        "workflow.step.geometry.label": "形状/CAD",
        "workflow.step.geometry.next": "形状線、CAD、またはメッシュ生成条件を確認します。",
        "workflow.step.geometry.detail": "形状定義またはメッシュ生成条件があれば先へ進めます。",
        "workflow.step.mesh.label": "メッシュ",
        "workflow.step.mesh.next": "メッシュ生成条件、要素種別、分割数を設定します。",
        "workflow.step.mesh.detail": "メッシュ生成条件と要素種別を確認します。",
        "workflow.step.materials.label": "材料",
        "workflow.step.materials.next": "材料モデルと主要パラメータを設定します。",
        "workflow.step.materials.detail": "材料モデル、E、nuなどを確認します。",
        "workflow.step.boundary_conditions.label": "境界条件",
        "workflow.step.boundary_conditions.next": "拘束条件またはステージ内境界条件を設定します。",
        "workflow.step.boundary_conditions.detail": "拘束条件を設定します。",
        "workflow.step.loads.label": "荷重",
        "workflow.step.loads.next": "荷重、自己重量、水圧、ステージ荷重を必要に応じて設定します。",
        "workflow.step.loads.detail": "荷重が不要な解析では省略できます。",
        "workflow.step.stages.label": "ステージ",
        "workflow.step.stages.next": "施工段階、材料変更、応力解放率を必要に応じて設定します。",
        "workflow.step.stages.detail": "単一ステージ解析では省略できます。",
        "workflow.step.model_check.label": "モデルチェック",
        "workflow.step.model_check.next": "解析前チェックを実行し、エラーを解消します。",
        "workflow.step.model_check.detail": "解析前チェックを実行します。",
        "workflow.step.solver.label": "解析実行",
        "workflow.step.solver.next": "解析を実行し、収束条件と出力先を確認します。",
        "workflow.step.solver.detail": "モデルチェック後に解析を実行します。",
        "workflow.step.results.label": "結果確認",
        "workflow.step.results.next": "変位、応力、FL、履歴などの結果を確認します。",
        "workflow.step.results.detail": "解析後に結果ファイルを確認します。",
        "workflow.step.report.label": "帳票",
        "workflow.step.report.next": "HTML/PDF帳票とmanifestを確認します。",
        "workflow.step.report.detail": "帳票生成後にHTML/PDFとmanifestを確認します。",
        "workflow.completed": "完了しています。",
        "workflow.status.complete": "完了",
        "workflow.status.missing": "未入力",
        "workflow.status.optional": "任意",
        "workflow.required": "必須",
        "workflow.optional": "任意",
        "workflow.progress": "必須入力充足率: {ratio:.1f}% / 次の操作: {next_label} - {next_action}",
        "workflow.progress_ratio": "必須入力充足率: {ratio:.1f}%",
        "workflow.table.step": "工程",
        "workflow.table.role": "区分",
        "workflow.table.required": "必須",
        "workflow.table.status": "状態",
        "workflow.table.missing": "不足箇所",
        "workflow.table.next": "次操作",
        "workflow.table.jump": "移動先",
        "workflow.default_next_label": "出力確認",
        "workflow.default_next_action": "必要な追加作業はありません。",
        "workflow.refresh_button": "ワークフロー診断を更新",
        "workflow.jump_status": "ワークフロー移動: {label} / {next_action}",
        "workflow.report.title": "ワークフローガイダンス",
        "command.report.title": "GUIコマンドカタログ",
        "command.table.id": "ID",
        "command.table.label": "表示名",
        "command.table.role": "役割",
        "command.table.target": "移動先",
        "command.table.shortcut": "ショートカット",
        "command.table.context": "右クリック",
        "command.table.tooltip": "説明",
    },
    "en": {
        "gui.language.ja": "Japanese",
        "gui.language.en": "English",
        "gui.menu.operations": "Operations",
        "gui.menu.advanced_settings": "Advanced Settings",
        "gui.menu.language": "Language",
        "gui.language.switch_to_ja": "Switch to Japanese",
        "gui.language.switch_to_en": "Switch to English",
        "gui.toolbar.main": "Main Actions",
        "gui.group.primary": "Run",
        "gui.group.confirm": "Check",
        "gui.group.output": "Output",
        "gui.group.detail": "Advanced",
        "gui.group.project": "Project",
        "gui.label.undo_granularity": "Undo Scope",
        "gui.action_role.primary": "Main Action",
        "gui.action_role.confirm": "Check",
        "gui.action_role.output": "Output",
        "gui.action_role.detail": "Advanced",
        "gui.command.palette.label": "Command Palette...",
        "gui.command.palette.tooltip": "Choose and run a command in a popup.",
        "gui.command.workflow.open.label": "Workflow",
        "gui.command.workflow.open.tooltip": "Show the overall workflow and missing inputs.",
        "gui.command.input.save.label": "Save",
        "gui.command.input.save.tooltip": "Save the current input.",
        "gui.command.analysis.run.label": "Run Analysis",
        "gui.command.analysis.run.tooltip": "Run the analysis after model checks.",
        "gui.command.workflow.refresh.label": "Refresh Workflow Check",
        "gui.command.workflow.refresh.tooltip": "Update required-input completion and next action.",
        "gui.command.model.check.label": "Model Check",
        "gui.command.model.check.tooltip": "Check input errors and warnings before analysis.",
        "gui.command.analysis.stop.label": "Stop",
        "gui.command.analysis.stop.tooltip": "Stop the running analysis.",
        "gui.command.errors.jump.label": "Jump to Error Cell",
        "gui.command.errors.jump.tooltip": "Move to the selected input error location.",
        "gui.command.errors.fix_all.label": "Fix All Errors",
        "gui.command.errors.fix_all.tooltip": "Apply available automatic fixes to input errors.",
        "gui.command.results.open.label": "Results",
        "gui.command.results.open.tooltip": "Open the latest analysis results.",
        "gui.command.report.open.label": "Report",
        "gui.command.report.open.tooltip": "Check report output and manifest.",
        "gui.command.workspace.dashboard.label": "Workspace Dashboard",
        "gui.command.workspace.dashboard.tooltip": "Write project, recent run, artifact, storage, and archive planning artifacts.",
        "gui.command.input.template.load.label": "Load Input Template",
        "gui.command.input.template.load.tooltip": "Load an input template that can include geometry, materials, boundaries, loads, and stages.",
        "gui.command.input.sync.label": "Apply YAML",
        "gui.command.input.sync.tooltip": "Apply edited YAML to the forms.",
        "gui.command.customization.apply.label": "Apply Organization Profile",
        "gui.command.customization.apply.tooltip": "Apply organization logo, report template, units, colors, and defaults to the current input.",
        "gui.command.yaml.open.label": "Open YAML",
        "gui.command.yaml.open.tooltip": "Open the detailed input YAML.",
        "gui.command.diagnostics.recovery.label": "Recovery Candidates",
        "gui.command.diagnostics.recovery.tooltip": "Inspect autosave recovery candidates.",
        "gui.command.diagnostics.audit.label": "Audit Log",
        "gui.command.diagnostics.audit.tooltip": "Inspect the operation audit log.",
        "workflow.step.analysis.label": "Analysis",
        "workflow.step.analysis.next": "Check analysis type, dimension, and unit system.",
        "workflow.step.analysis.detail": "Check the analysis type and 2D setting.",
        "workflow.step.geometry.label": "Geometry/CAD",
        "workflow.step.geometry.next": "Check geometry lines, CAD, or mesh generation settings.",
        "workflow.step.geometry.detail": "Geometry or mesh generation settings are enough to proceed.",
        "workflow.step.mesh.label": "Mesh",
        "workflow.step.mesh.next": "Set mesh generation, element type, and divisions.",
        "workflow.step.mesh.detail": "Check mesh generation settings and element type.",
        "workflow.step.materials.label": "Materials",
        "workflow.step.materials.next": "Set material models and primary parameters.",
        "workflow.step.materials.detail": "Check material model, E, nu, and related parameters.",
        "workflow.step.boundary_conditions.label": "Boundary Conditions",
        "workflow.step.boundary_conditions.next": "Set constraints or stage boundary conditions.",
        "workflow.step.boundary_conditions.detail": "Set constraints.",
        "workflow.step.loads.label": "Loads",
        "workflow.step.loads.next": "Set loads, self weight, water pressure, or stage loads as needed.",
        "workflow.step.loads.detail": "Loads can be omitted when the analysis does not need them.",
        "workflow.step.stages.label": "Stages",
        "workflow.step.stages.next": "Set construction stages, material changes, or stress release as needed.",
        "workflow.step.stages.detail": "Stages can be omitted for a single-stage analysis.",
        "workflow.step.model_check.label": "Model Check",
        "workflow.step.model_check.next": "Run pre-analysis checks and resolve ERROR items.",
        "workflow.step.model_check.detail": "Run the pre-analysis model check.",
        "workflow.step.solver.label": "Run",
        "workflow.step.solver.next": "Run the analysis and check convergence/output settings.",
        "workflow.step.solver.detail": "Run the analysis after model checks.",
        "workflow.step.results.label": "Results",
        "workflow.step.results.next": "Review displacement, stress, FL, and history results.",
        "workflow.step.results.detail": "Review result files after analysis.",
        "workflow.step.report.label": "Report",
        "workflow.step.report.next": "Check HTML/PDF reports and manifest.",
        "workflow.step.report.detail": "Check HTML/PDF reports and manifest after report generation.",
        "workflow.completed": "Completed.",
        "workflow.status.complete": "Complete",
        "workflow.status.missing": "Missing",
        "workflow.status.optional": "Optional",
        "workflow.required": "Required",
        "workflow.optional": "Optional",
        "workflow.progress": "Required input completion: {ratio:.1f}% / Next action: {next_label} - {next_action}",
        "workflow.progress_ratio": "Required input completion: {ratio:.1f}%",
        "workflow.table.step": "Step",
        "workflow.table.role": "Role",
        "workflow.table.required": "Required",
        "workflow.table.status": "Status",
        "workflow.table.missing": "Missing Paths",
        "workflow.table.next": "Next Action",
        "workflow.table.jump": "Jump Target",
        "workflow.default_next_label": "Output Review",
        "workflow.default_next_action": "No additional required action.",
        "workflow.refresh_button": "Refresh Workflow Check",
        "workflow.jump_status": "Workflow jump: {label} / {next_action}",
        "workflow.report.title": "Workflow Guidance",
        "command.report.title": "GUI Command Catalog",
        "command.table.id": "ID",
        "command.table.label": "Label",
        "command.table.role": "Role",
        "command.table.target": "Target",
        "command.table.shortcut": "Shortcut",
        "command.table.context": "Context Menu",
        "command.table.tooltip": "Description",
    },
}


def gui_message(message_key: str, *, locale: str = DEFAULT_GUI_LOCALE, **values: Any) -> str:
    table = GUI_MESSAGES.get(locale, GUI_MESSAGES[DEFAULT_GUI_LOCALE])
    text = table.get(message_key, GUI_MESSAGES[DEFAULT_GUI_LOCALE].get(message_key, message_key))
    if values:
        try:
            return text.format(**values)
        except Exception:
            return text
    return text


def gui_message_catalog(locale: str = DEFAULT_GUI_LOCALE) -> dict[str, str]:
    return dict(GUI_MESSAGES.get(locale, GUI_MESSAGES[DEFAULT_GUI_LOCALE]))


def available_gui_message_keys(locale: str = DEFAULT_GUI_LOCALE) -> list[str]:
    return sorted(gui_message_catalog(locale))


def validate_gui_i18n_catalog(required_keys: Iterable[str] | None = None) -> dict[str, Any]:
    keys_by_locale = {locale: set(gui_message_catalog(locale)) for locale in SUPPORTED_GUI_LOCALES}
    baseline = keys_by_locale[DEFAULT_GUI_LOCALE]
    expected = set(required_keys or baseline)
    missing = {
        locale: sorted((baseline | expected) - keys)
        for locale, keys in keys_by_locale.items()
    }
    extra = {
        locale: sorted(keys - (baseline | expected))
        for locale, keys in keys_by_locale.items()
    }
    errors = []
    for locale, keys in missing.items():
        if keys:
            errors.append(f"{locale} missing: {', '.join(keys)}")
    return {
        "schema": "geofem.gui.i18n.validation.v1",
        "passed": not errors,
        "errors": errors,
        "missing": missing,
        "extra": extra,
        "locales": list(SUPPORTED_GUI_LOCALES),
        "key_count": len(baseline | expected),
    }


def write_gui_i18n_catalog(output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "gui_i18n_catalog.json"
    csv_path = out / "gui_i18n_catalog.csv"
    validation_path = out / "gui_i18n_validation.json"
    json_path.write_text(json.dumps(GUI_MESSAGES, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted(set().union(*(set(gui_message_catalog(locale)) for locale in SUPPORTED_GUI_LOCALES)))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", *SUPPORTED_GUI_LOCALES])
        for key in keys:
            writer.writerow([key, *(gui_message(key, locale=locale) for locale in SUPPORTED_GUI_LOCALES)])
    validation = validate_gui_i18n_catalog(keys)
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "validation": str(validation_path)}


__all__ = [
    "DEFAULT_GUI_LOCALE",
    "GUI_MESSAGES",
    "SUPPORTED_GUI_LOCALES",
    "available_gui_message_keys",
    "gui_message",
    "gui_message_catalog",
    "validate_gui_i18n_catalog",
    "write_gui_i18n_catalog",
]
