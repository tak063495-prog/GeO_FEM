"""Localized visible GUI surface texts beyond the shared command catalog."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .i18n import DEFAULT_GUI_LOCALE, SUPPORTED_GUI_LOCALES


GUI_SURFACE_TEXT_SCHEMA = "geofem.gui.surface_text_catalog.v1"
GUI_SURFACE_TEXT_VALIDATION_SCHEMA = "geofem.gui.surface_text_validation.v1"
REQUIRED_SURFACE_CATEGORIES = {"button", "form", "log", "menu", "panel", "report", "result", "status", "tab", "tree"}
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class SurfaceText:
    key: str
    category: str
    ja: str
    en: str


SURFACE_TEXTS: tuple[SurfaceText, ...] = (
    SurfaceText("tree.header.project", "tree", "プロジェクト", "Project"),
    SurfaceText("tree.root", "tree", "GeoFEM 2D", "GeoFEM 2D"),
    SurfaceText("tree.group.workflow", "tree", "GeoFEAS操作ワークフロー", "GeoFEAS Workflow"),
    SurfaceText("tree.group.workflow_flow", "tree", "GeoFEAS操作フロー", "GeoFEAS Workflow"),
    SurfaceText("tree.group.pre", "tree", "プリプロセッサ", "Preprocessor"),
    SurfaceText("tree.group.post", "tree", "ポストプロセッサ", "Postprocessor"),
    SurfaceText("tree.group.advanced", "tree", "上級者", "Advanced"),
    SurfaceText("tree.group.basic_input", "tree", "基本入力", "Basic Input"),
    SurfaceText("tree.group.check_output", "tree", "確認/出力", "Check/Output"),
    SurfaceText("tree.group.check", "tree", "確認", "Check"),
    SurfaceText("tree.group.execution", "tree", "実行", "Execution"),
    SurfaceText("tree.group.output", "tree", "出力", "Output"),
    SurfaceText("tree.item.workflow_guide", "tree", "作業ガイド", "Workflow Guide"),
    SurfaceText("panel.workflow", "panel", "GeoFEAS操作", "GeoFEAS Workflow"),
    SurfaceText("panel.analysis", "panel", "解析条件", "Analysis"),
    SurfaceText("panel.geometry", "panel", "形状/CAD", "Geometry/CAD"),
    SurfaceText("panel.mesh", "panel", "メッシュ", "Mesh"),
    SurfaceText("panel.mesh_split", "panel", "メッシュ分割", "Mesh Division"),
    SurfaceText("panel.materials", "panel", "材料", "Materials"),
    SurfaceText("panel.external", "panel", "外部連携", "External Linkage"),
    SurfaceText("panel.boundary", "panel", "境界条件", "Boundary Conditions"),
    SurfaceText("panel.loads", "panel", "荷重", "Loads"),
    SurfaceText("panel.stages", "panel", "ステージ", "Stages"),
    SurfaceText("panel.stage_settings", "panel", "ステージ設定", "Stage Settings"),
    SurfaceText("panel.solver", "panel", "ソルバ", "Solver"),
    SurfaceText("panel.model_check", "panel", "モデルチェック", "Model Check"),
    SurfaceText("panel.confirm", "panel", "確認", "Check"),
    SurfaceText("panel.results", "panel", "結果", "Results"),
    SurfaceText("panel.results_check", "panel", "結果確認", "Result Review"),
    SurfaceText("panel.result_stress_contour", "tree", "応力コンター", "Stress Contour"),
    SurfaceText("panel.result_displacement_vector", "tree", "変位ベクトル", "Displacement Vectors"),
    SurfaceText("panel.result_displacement_contour", "tree", "変位コンタ", "Displacement Contour"),
    SurfaceText("panel.result_displacement_table", "tree", "変形図/変位表", "Deformed/Table"),
    SurfaceText("panel.result_pore_pressure", "tree", "水圧コンタ", "Pore Pressure"),
    SurfaceText("panel.result_safety_factor", "tree", "FL/安全率", "FL/Safety Factor"),
    SurfaceText("panel.result_convergence", "tree", "収束履歴", "Convergence History"),
    SurfaceText("panel.result_performance", "tree", "性能履歴", "Performance History"),
    SurfaceText("panel.report", "panel", "計算書", "Report"),
    SurfaceText("panel.report_create", "panel", "計算書作成", "Report Builder"),
    SurfaceText("panel.data_save", "panel", "データ保存", "Data Save"),
    SurfaceText("panel.yaml", "panel", "YAML", "YAML"),
    SurfaceText("label.operation_mode", "form", "操作モード", "Operation Mode"),
    SurfaceText("label.command_search", "form", "コマンド検索", "Command Search"),
    SurfaceText("label.workflow", "form", "ワークフロー", "Workflow"),
    SurfaceText("label.current", "form", "現在", "Current"),
    SurfaceText("label.quality", "form", "品質", "Quality"),
    SurfaceText("label.limit", "form", "上限", "Limit"),
    SurfaceText("label.set", "form", "セット", "Set"),
    SurfaceText("label.selection_operation", "form", "選択操作", "Selection"),
    SurfaceText("label.snap_target", "form", "吸着", "Snap To"),
    SurfaceText("label.grid_spacing", "form", "格子", "Grid"),
    SurfaceText("group.status", "form", "状況", "Status"),
    SurfaceText("group.next_action", "form", "次の操作", "Next Action"),
    SurfaceText("group.result_display_adjustment", "form", "結果表示調整", "Result Display"),
    SurfaceText("label.result_stage", "form", "結果ステージ", "Result Stage"),
    SurfaceText("label.result_table_component", "form", "表/分布成分", "Table / Distribution Component"),
    SurfaceText("label.result_table_empty", "form", "数値表: 0件", "Value Table: 0 rows"),
    SurfaceText("label.report_summary", "form", "解析サマリ", "Analysis Summary"),
    SurfaceText("label.report_table_links", "form", "数値表リンク", "Value Table Links"),
    SurfaceText("label.result_stage_time", "form", "ステージ/時刻", "Stage/Time"),
    SurfaceText("label.result_time_history", "form", "時間経過", "Time"),
    SurfaceText("label.deformation_scale", "form", "変形倍率", "Deformation Scale"),
    SurfaceText("label.result_component", "form", "表示成分", "Component"),
    SurfaceText("label.result_colormap", "form", "色表示", "Color Map"),
    SurfaceText("label.stage_analysis_state", "form", "解析状態", "Analysis State"),
    SurfaceText("label.stage_name", "form", "名称", "Name"),
    SurfaceText("label.stage_action", "form", "操作", "Action"),
    SurfaceText("label.stage_action_condition", "form", "操作・条件", "Action / Condition"),
    SurfaceText("label.stage_boundary", "form", "境界", "Boundary"),
    SurfaceText("label.stage_hydraulic", "form", "水理", "Hydraulic"),
    SurfaceText("label.stage_mpc", "form", "多点拘束", "MPC"),
    SurfaceText("label.stage_node_set", "form", "節点集合", "Node Set"),
    SurfaceText("label.stage_x_displacement", "form", "X変位", "X Displacement"),
    SurfaceText("label.stage_y_displacement", "form", "Y変位", "Y Displacement"),
    SurfaceText("label.stage_x_load", "form", "X荷重", "X Load"),
    SurfaceText("label.stage_y_load", "form", "Y荷重", "Y Load"),
    SurfaceText("label.stage_x_distributed_load", "form", "X分布荷重", "X Distributed Load"),
    SurfaceText("label.stage_y_distributed_load", "form", "Y分布荷重", "Y Distributed Load"),
    SurfaceText("label.stage_stress_release", "form", "応力解放", "Stress Release"),
    SurfaceText("label.stage_stress_release_ratio", "form", "応力解放率", "Stress Release Ratio"),
    SurfaceText("label.stage_reactivate", "form", "再有効化", "Reactivate"),
    SurfaceText("label.stage_analysis_details", "form", "解析詳細設定", "Analysis Details"),
    SurfaceText("label.stage_overrides", "form", "個別設定", "Overrides"),
    SurfaceText("label.stage_additional_settings", "form", "追加設定", "Additional Settings"),
    SurfaceText("result.judgment_summary", "result", "判定サマリ", "Judgment Summary"),
    SurfaceText("result.empty_title", "result", "解析結果はまだありません", "No analysis results yet"),
    SurfaceText(
        "result.empty_detail",
        "result",
        "解析を実行すると、ここに判定、主要値、探索精度、解析時間を表示します。",
        "Run an analysis to show its judgment, key values, search precision, and elapsed time here.",
    ),
    SurfaceText("tab.model", "tab", "モデル", "Model"),
    SurfaceText("tab.check", "tab", "チェック", "Check"),
    SurfaceText("tab.input_assist", "tab", "入力補助", "Input Assist"),
    SurfaceText("tab.input_form", "tab", "入力フォーム", "Input Forms"),
    SurfaceText("tab.result", "tab", "結果", "Results"),
    SurfaceText("tab.table", "tab", "数値表", "Value Table"),
    SurfaceText("tab.selection_history", "tab", "選択履歴", "Selection History"),
    SurfaceText("tab.named_selection", "tab", "名前付き選択", "Named Selections"),
    SurfaceText("tab.selection_compare", "tab", "選択比較", "Selection Compare"),
    SurfaceText("tab.input_error", "tab", "入力エラー", "Input Errors"),
    SurfaceText("tab.recovery", "tab", "復旧候補", "Recovery"),
    SurfaceText("tab.audit", "tab", "監査ログ", "Audit Log"),
    SurfaceText("tab.stage_detail", "tab", "詳細フォーム", "Detail Form"),
    SurfaceText("tab.stage_diff", "tab", "差分/承認", "Diff/Approval"),
    SurfaceText("tab.stage_internal_data", "tab", "内部データ", "Internal Data"),
    SurfaceText("tab.material_list", "tab", "材料一覧", "Materials"),
    SurfaceText("tab.material_library", "tab", "材料ライブラリ", "Material Library"),
    SurfaceText("tab.result_display", "tab", "表示設定", "Display"),
    SurfaceText("tab.result_srm", "tab", "SRM", "SRM"),
    SurfaceText("tab.result_output", "tab", "出力・比較", "Output / Compare"),
    SurfaceText("menu.file", "menu", "ファイル", "File"),
    SurfaceText("menu.edit", "menu", "編集", "Edit"),
    SurfaceText("menu.analysis", "menu", "解析", "Analysis"),
    SurfaceText("menu.selection", "menu", "選択", "Selection"),
    SurfaceText("menu.view", "menu", "表示", "View"),
    SurfaceText("menu.operations", "menu", "操作", "Operations"),
    SurfaceText("menu.advanced_settings", "menu", "詳細設定", "Advanced Settings"),
    SurfaceText("menu.language", "menu", "言語", "Language"),
    SurfaceText("menu.help", "menu", "ヘルプ", "Help"),
    SurfaceText("menu.recent", "menu", "最近使ったプロジェクト", "Recent Projects"),
    SurfaceText("menu.recent_empty", "menu", "履歴なし", "No History"),
    SurfaceText("menu.import", "menu", "インポート", "Import"),
    SurfaceText("menu.export", "menu", "エクスポート", "Export"),
    SurfaceText("action.new_sample", "button", "新規2Dサンプル", "New 2D Sample"),
    SurfaceText("action.open_input", "button", "入力を開く", "Open Input"),
    SurfaceText("action.save_input", "button", "入力を保存", "Save Input"),
    SurfaceText("action.save_input_as", "button", "名前を付けて保存", "Save As"),
    SurfaceText("action.navigation_pane", "button", "ナビゲーションペイン", "Navigation Pane"),
    SurfaceText("action.inspector_pane", "button", "インスペクタ", "Inspector"),
    SurfaceText("action.application_log", "button", "ログ", "Log"),
    SurfaceText("action.workspace_focus", "button", "表示を広げる", "Expand View"),
    SurfaceText("action.workspace_focus_restore", "button", "通常表示", "Restore Layout"),
    SurfaceText("action.import_cad", "button", "CAD線形をインポート", "Import CAD Lines"),
    SurfaceText("action.import_pore_pressure", "button", "浸透CSV水圧をインポート", "Import Seepage CSV"),
    SurfaceText("action.import_waterline", "button", "水位線CSVをインポート", "Import Waterline CSV"),
    SurfaceText("action.import_project_template", "button", "入力テンプレートを読込", "Load Input Template"),
    SurfaceText("action.export_sxf", "button", "形状SXFをエクスポート", "Export Geometry SXF"),
    SurfaceText("action.export_result_csv", "button", "結果表CSVをエクスポート", "Export Result CSV"),
    SurfaceText("action.export_scene_image", "button", "図面画像をエクスポート", "Export Scene Image"),
    SurfaceText("action.export_scene_pdf", "button", "図面PDFをエクスポート", "Export Scene PDF"),
    SurfaceText("action.export_report_pdf", "button", "計算書PDFをエクスポート", "Export Report PDF"),
    SurfaceText("action.exit", "button", "終了", "Exit"),
    SurfaceText("action.undo", "button", "編集を元に戻す", "Undo Edit"),
    SurfaceText("action.redo", "button", "編集をやり直す", "Redo Edit"),
    SurfaceText("action.cut", "button", "切り取り", "Cut"),
    SurfaceText("action.copy", "button", "コピー", "Copy"),
    SurfaceText("action.paste", "button", "貼り付け", "Paste"),
    SurfaceText("action.delete", "button", "削除", "Delete"),
    SurfaceText("action.coordinate_edit", "button", "座標編集", "Edit Coordinates"),
    SurfaceText("action.pan", "button", "パン", "Pan"),
    SurfaceText("action.panel_display", "button", "パネル表示", "Panels"),
    SurfaceText("action.quality_cycle", "button", "表示品質切替", "Cycle Quality"),
    SurfaceText("action.result_legend", "button", "結果凡例", "Result Legend"),
    SurfaceText("action.analysis_conditions", "button", "解析条件", "Analysis Conditions"),
    SurfaceText("action.reset_results", "button", "結果リセット", "Reset Results"),
    SurfaceText("action.convergence_log", "button", "収束ログ", "Convergence Log"),
    SurfaceText("action.refresh_results", "button", "結果更新", "Refresh Results"),
    SurfaceText("action.riks_path", "button", "Riks経路", "Riks Path"),
    SurfaceText("action.large_model_operations", "button", "大規模操作", "Large Model Operations"),
    SurfaceText("action.find_node", "button", "節点検索", "Find Node"),
    SurfaceText("action.find_element", "button", "要素検索", "Find Element"),
    SurfaceText("action.srm_summary", "button", "SRM要約", "SRM Summary"),
    SurfaceText("action.open_result_figure", "button", "結果図を開く", "Open Result Figure"),
    SurfaceText("action.open_report_step", "button", "帳票へ進む", "Open Report"),
    SurfaceText("action.open_solver_step", "button", "解析実行へ進む", "Open Analysis Run"),
    SurfaceText("action.show_result_details", "button", "詳細な結果を表示", "Show Detailed Results"),
    SurfaceText("action.hide_result_details", "button", "詳細な結果を隠す", "Hide Detailed Results"),
    SurfaceText("action.distribution_plot", "button", "分布図", "Distribution Plot"),
    SurfaceText("action.line_distribution", "button", "任意測線分布", "Line Distribution"),
    SurfaceText("action.duplicate_post_view", "button", "Postビュー複製", "Duplicate Post View"),
    SurfaceText("action.add_drawing_layout", "button", "図面配置へ追加", "Add to Drawing Layout"),
    SurfaceText("action.save_drawing_frame", "button", "図面枠保存", "Save Drawing Frame"),
    SurfaceText("action.load_drawing_frame", "button", "図面枠読込", "Load Drawing Frame"),
    SurfaceText("action.save_post_baseline", "button", "Post基準保存", "Save Post Baseline"),
    SurfaceText("action.compare_post_image", "button", "Post画像差分", "Compare Post Image"),
    SurfaceText("action.image_diff_ci", "button", "画像差分CI", "Image Diff CI"),
    SurfaceText("action.pdf_drawing", "button", "PDF図面", "PDF Drawing"),
    SurfaceText("action.preview_existing_report", "button", "既存レポートをプレビュー", "Preview Existing Report"),
    SurfaceText("action.audit_post_report", "button", "Post/帳票監査", "Audit Post / Report"),
    SurfaceText("action.preview_wysiwyg_report", "button", "WYSIWYG帳票プレビュー", "Preview WYSIWYG Report"),
    SurfaceText("action.version", "button", "バージョン情報", "Version Info"),
    SurfaceText("action.lasso_select", "button", "投げ縄選択", "Lasso Select"),
    SurfaceText("action.invert_selection", "button", "選択反転", "Invert Selection"),
    SurfaceText("action.undo_selection", "button", "選択を元に戻す", "Undo Selection"),
    SurfaceText("action.redo_selection", "button", "選択をやり直す", "Redo Selection"),
    SurfaceText("action.open_selection_history", "button", "選択履歴を開く", "Open Selection History"),
    SurfaceText("button.fit", "button", "全体表示", "Fit"),
    SurfaceText("button.back", "button", "戻る", "Back"),
    SurfaceText("button.next_move", "button", "次へ移動", "Next"),
    SurfaceText("button.model_check", "button", "モデルチェック", "Model Check"),
    SurfaceText("button.load", "button", "読込", "Load"),
    SurfaceText("button.run_input", "button", "入力実行", "Run Input"),
    SurfaceText("button.refresh_diagnostics", "button", "診断更新", "Refresh Diagnostics"),
    SurfaceText("button.draw_line", "button", "線作図", "Draw Line"),
    SurfaceText("button.select", "button", "選択", "Select"),
    SurfaceText("button.part_select", "button", "部分選択", "Part Select"),
    SurfaceText("button.straight_line", "button", "直線", "Line"),
    SurfaceText("button.polyline", "button", "折れ線", "Polyline"),
    SurfaceText("button.rectangle", "button", "矩形", "Rectangle"),
    SurfaceText("button.polygon", "button", "多角形", "Polygon"),
    SurfaceText("button.circle", "button", "円", "Circle"),
    SurfaceText("button.arc", "button", "円弧", "Arc"),
    SurfaceText("button.curve", "button", "曲線", "Curve"),
    SurfaceText("button.point", "button", "点", "Point"),
    SurfaceText("button.eraser", "button", "消しゴム", "Eraser"),
    SurfaceText("button.move", "button", "移動", "Move"),
    SurfaceText("button.trim", "button", "トリム", "Trim"),
    SurfaceText("button.extend", "button", "延長", "Extend"),
    SurfaceText("button.split", "button", "分割", "Split"),
    SurfaceText("button.helper_line", "button", "補助線", "Helper Line"),
    SurfaceText("button.grid", "button", "グリッド", "Grid"),
    SurfaceText("button.mesh_control_point", "button", "制御点", "Control Point"),
    SurfaceText("button.mesh_size", "button", "サイズ", "Size"),
    SurfaceText("button.mesh_block_split", "button", "ブロック分割", "Block Split"),
    SurfaceText("button.violation_select", "button", "違反選択", "Select Violations"),
    SurfaceText("button.mesh_delete_short", "button", "削除", "Delete"),
    SurfaceText("button.draw_region", "button", "領域作図", "Draw Region"),
    SurfaceText("button.close_region", "button", "領域を閉じる", "Close Region"),
    SurfaceText("button.snap", "button", "スナップ", "Snap"),
    SurfaceText("button.node_id", "button", "節点ID", "Node ID"),
    SurfaceText("button.element_id", "button", "要素ID", "Element ID"),
    SurfaceText("button.rectangle_select", "button", "矩形選択", "Box Select"),
    SurfaceText("button.lasso", "button", "投げ縄", "Lasso"),
    SurfaceText("button.filter_select", "button", "条件選択", "Filter Select"),
    SurfaceText("button.clear_select", "button", "選択解除", "Clear Selection"),
    SurfaceText("button.invert", "button", "反転", "Invert"),
    SurfaceText("button.history", "button", "履歴記録", "Record History"),
    SurfaceText("button.name_save", "button", "名前保存", "Save Name"),
    SurfaceText("button.selection_compare", "button", "選択比較", "Compare Selection"),
    SurfaceText("button.mode_help", "button", "モードヘルプ", "Mode Help"),
    SurfaceText("button.selection_set", "button", "選択set登録", "Save Selection Set"),
    SurfaceText("button.run", "button", "解析実行", "Run"),
    SurfaceText("button.stop", "button", "停止", "Stop"),
    SurfaceText("button.sync_yaml", "button", "YAMLから反映", "Apply"),
    SurfaceText("button.save", "button", "保存", "Save"),
    SurfaceText("button.autosave", "button", "自動保存", "Autosave"),
    SurfaceText("button.execute", "button", "実行", "Run"),
    SurfaceText("button.execute_command", "button", "コマンド実行", "Run Command"),
    SurfaceText("button.pin", "button", "ピン", "Pin"),
    SurfaceText("button.open_detail", "button", "詳細を開く", "Open Detail"),
    SurfaceText("button.undo_point", "button", "戻し点", "Undo Point"),
    SurfaceText("button.lock", "button", "ロック", "Lock"),
    SurfaceText("button.unlock", "button", "ロック解除", "Unlock"),
    SurfaceText("button.force_unlock", "button", "強制解除", "Force"),
    SurfaceText("button.handoff", "button", "引継ぎ", "Handoff"),
    SurfaceText("button.jump_error", "button", "エラーセルへ", "Jump"),
    SurfaceText("button.fix_errors", "button", "エラー一括修正", "Fix All"),
    SurfaceText("button.recovery", "button", "復旧候補", "Recover"),
    SurfaceText("button.audit", "button", "監査ログ", "Audit"),
    SurfaceText("button.apply_rectangle_mesh", "button", "矩形メッシュを反映", "Apply Rectangle Mesh"),
    SurfaceText("button.apply_geometry", "button", "形状表を反映", "Apply Geometry Table"),
    SurfaceText("button.move_to_coordinates", "button", "座標へ移動", "Move to Coordinates"),
    SurfaceText("button.relative_move", "button", "相対移動", "Relative Move"),
    SurfaceText("button.join_endpoints", "button", "端点結合", "Join Endpoints"),
    SurfaceText("button.close_polyline", "button", "閉合線", "Close Loop"),
    SurfaceText("button.mark_construction", "button", "補助線化", "Make Construction"),
    SurfaceText("button.mark_split", "button", "分割線化", "Make Split Line"),
    SurfaceText("button.make_region", "button", "領域化", "Create Region"),
    SurfaceText("button.grid_condition", "button", "格子条件", "Grid Condition"),
    SurfaceText("button.apply_conditions", "button", "条件反映", "Apply Conditions"),
    SurfaceText("button.apply_controls", "button", "制御反映", "Apply Controls"),
    SurfaceText("button.rebuild", "button", "再構成", "Rebuild"),
    SurfaceText("button.delete_mesh", "button", "メッシュ削除", "Delete Mesh"),
    SurfaceText("button.add_control_point", "button", "制御点追加", "Add Control Point"),
    SurfaceText("button.delete_selected", "button", "選択削除", "Delete Selected"),
    SurfaceText("button.delete_selected_rows", "button", "選択行削除", "Delete Selected Rows"),
    SurfaceText("button.local_refine", "button", "局所細分", "Local Refine"),
    SurfaceText("button.size_range", "button", "サイズ範囲", "Size Range"),
    SurfaceText("button.split_line", "button", "分割線", "Split Line"),
    SurfaceText("button.extract_quality", "button", "品質抽出", "Extract Quality"),
    SurfaceText("button.repair_violation", "button", "違反修復", "Repair Violations"),
    SurfaceText("button.add", "button", "追加", "Add"),
    SurfaceText("button.elastic", "button", "弾性", "Elastic"),
    SurfaceText("button.nonlinear_elastic", "button", "非線形弾性", "Nonlinear Elastic"),
    SurfaceText("button.liquefaction", "button", "液状化", "Liquefaction"),
    SurfaceText("button.remove", "button", "削除", "Delete"),
    SurfaceText("button.apply_materials", "button", "材料を反映", "Apply Materials"),
    SurfaceText("button.fixed", "button", "固定", "Fixed"),
    SurfaceText("button.prescribed_displacement", "button", "強制変位", "Prescribed Displacement"),
    SurfaceText("button.horizontal_roller", "button", "水平ローラ", "Horizontal Roller"),
    SurfaceText("button.vertical_roller", "button", "鉛直ローラ", "Vertical Roller"),
    SurfaceText("button.apply_table", "button", "表を反映", "Apply Table"),
    SurfaceText("button.apply_boundary_table", "button", "境界条件表を反映", "Apply Boundary Table"),
    SurfaceText("button.selected_nodes_to_support", "button", "選択節点→支点/変位", "Selected Nodes -> Support/Displacement"),
    SurfaceText("button.selected_edges_to_hydro", "button", "選択辺/要素→水理境界", "Selected Edges/Elements -> Hydraulic Boundary"),
    SurfaceText("button.selected_nodes_to_mpc", "button", "選択節点→MPC", "Selected Nodes -> MPC"),
    SurfaceText("button.selected_nodes_to_set", "button", "選択節点→set登録", "Selected Nodes -> Save Set"),
    SurfaceText("button.apply_bc", "button", "境界条件YAMLを反映", "Apply Boundary YAML"),
    SurfaceText("button.add_static_case", "button", "静的ケース追加", "Add Static Case"),
    SurfaceText("button.add_seismic_case", "button", "地震ケース追加", "Add Seismic Case"),
    SurfaceText("button.delete_case", "button", "ケース削除", "Delete Case"),
    SurfaceText("button.delete_load", "button", "荷重削除", "Delete Load"),
    SurfaceText("button.apply_case", "button", "ケース反映", "Apply Case"),
    SurfaceText("button.apply_load_table", "button", "荷重表を反映", "Apply Load Table"),
    SurfaceText("button.nodal_force", "button", "節点集中", "Nodal Force"),
    SurfaceText("button.distributed_load", "button", "分布荷重", "Distributed Load"),
    SurfaceText("button.self_weight", "button", "自重", "Self Weight"),
    SurfaceText("button.pseudo_static_seismic", "button", "疑似静的地震荷重", "Pseudo-static Seismic"),
    SurfaceText("button.selected_nodes_to_nodal_load", "button", "選択節点→節点荷重", "Selected Nodes -> Nodal Load"),
    SurfaceText("button.selected_material_to_body_force", "button", "選択要素材料→体積力", "Selected Element Material -> Body Force"),
    SurfaceText("button.selected_edges_to_surface_load", "button", "選択辺/要素→面荷重", "Selected Edges/Elements -> Surface Load"),
    SurfaceText("button.selected_edges_to_tapered_load", "button", "選択辺/要素→偏分布面荷重", "Selected Edges/Elements -> Tapered Surface Load"),
    SurfaceText("button.seepage_csv_to_pressure", "button", "浸透流CSV→水圧", "Seepage CSV -> Water Pressure"),
    SurfaceText("button.apply_load", "button", "荷重YAMLを反映", "Apply Load YAML"),
    SurfaceText("button.move_up", "button", "上へ", "Up"),
    SurfaceText("button.move_down", "button", "下へ", "Down"),
    SurfaceText("button.add_death", "button", "無効化追加", "Add Deactivation"),
    SurfaceText("button.add_birth", "button", "再有効化追加", "Add Reactivation"),
    SurfaceText("button.add_material_change", "button", "材料変更追加", "Add Material Change"),
    SurfaceText("button.add_boundary_change", "button", "境界変更追加", "Add Boundary Change"),
    SurfaceText("button.add_load_change", "button", "荷重変更追加", "Add Load Change"),
    SurfaceText("button.delete_change_row", "button", "変更行削除", "Delete Change Row"),
    SurfaceText("button.apply_change_table", "button", "変更表を反映", "Apply Change Table"),
    SurfaceText("button.apply_detail", "button", "詳細を反映", "Apply Detail"),
    SurfaceText("button.add_excavation_death", "button", "掘削/death追加", "Add Excavation/Death"),
    SurfaceText("button.record_reactivation", "button", "再有効化記録", "Record Reactivation"),
    SurfaceText("button.apply_recommended", "button", "推奨値を反映", "Apply Recommended"),
    SurfaceText("button.k0_preset", "button", "K0プリセット", "K0 Preset"),
    SurfaceText("button.consolidation_preset", "button", "圧密プリセット", "Consolidation Preset"),
    SurfaceText("button.riks_preset", "button", "Riksプリセット", "Riks Preset"),
    SurfaceText("button.excavation_template", "button", "掘削テンプレート", "Excavation Template"),
    SurfaceText("button.boundary_switch_template", "button", "境界切替テンプレート", "Boundary Switch Template"),
    SurfaceText("button.hydro_switch_template", "button", "水理切替テンプレート", "Hydraulic Switch Template"),
    SurfaceText("button.check_diff", "button", "差分確認", "Check Diff"),
    SurfaceText("button.to_construction_set", "button", "施工setへ", "To Construction Set"),
    SurfaceText("button.selected_nodes_to_bulk_displacement", "button", "選択節点→一括変位", "Selected Nodes -> Bulk Displacement"),
    SurfaceText("button.selected_edges_to_distributed_load", "button", "選択辺→分布荷重", "Selected Edges -> Distributed Load"),
    SurfaceText("button.seismic_load", "button", "地震荷重", "Seismic Load"),
    SurfaceText("button.selected_to_water_pressure_link", "button", "選択→水圧連携", "Selection -> Water Pressure Link"),
    SurfaceText("button.stage_detail_center", "button", "詳細フォームを中央に表示", "Open Detail Form"),
    SurfaceText("button.stage_diff_center", "button", "差分/承認を中央に表示", "Open Diff/Approval"),
    SurfaceText("button.stage_yaml_center", "button", "YAMLを中央に表示", "Open YAML"),
    SurfaceText("button.apply_stage_detail", "button", "選択ステージ詳細を反映", "Apply Stage Detail"),
    SurfaceText("button.apply_stage_yaml", "button", "ステージYAMLを反映", "Apply Stage YAML"),
    SurfaceText("button.prev_page", "button", "前ページ", "Previous Page"),
    SurfaceText("button.next_page", "button", "次ページ", "Next Page"),
    SurfaceText("button.latest_results", "button", "最新結果フォルダを表示", "Show Latest Results"),
    SurfaceText("button.deformed", "button", "変形図/変位表", "Deformation/Table"),
    SurfaceText("button.disp_contour", "button", "変位コンタ", "Displacement Contour"),
    SurfaceText("button.disp_vector", "button", "変位ベクトル", "Displacement Vector"),
    SurfaceText("button.stress_contour", "button", "応力コンター", "Stress Contour"),
    SurfaceText("button.plastic_srm", "button", "塑性/SRM表示", "Plastic/SRM"),
    SurfaceText("button.safety", "button", "FL/安全率図", "FL/Safety"),
    SurfaceText("button.reactions", "button", "反力表", "Reactions"),
    SurfaceText("button.interface", "button", "界面状態", "Interface State"),
    SurfaceText("button.pore", "button", "水圧表", "Pore Pressure"),
    SurfaceText("button.quality", "button", "メッシュ品質", "Mesh Quality"),
    SurfaceText("button.analysis_log", "button", "解析ログ", "Analysis Log"),
    SurfaceText("button.performance", "button", "性能", "Performance"),
    SurfaceText("button.standard_report", "button", "標準帳票", "Standard Report"),
    SurfaceText("button.export_csv", "button", "表CSV保存", "Export CSV"),
    SurfaceText("button.drawing", "button", "図面出力", "Drawing Export"),
    SurfaceText("button.post_verify", "button", "Post図検証", "Post Check"),
    SurfaceText("button.company_template", "button", "企業様式配置", "Install Template"),
    SurfaceText("button.report_preview", "button", "帳票プレビュー", "Report Preview"),
    SurfaceText("button.report_create", "button", "選択内容で計算書作成", "Build Report"),
    SurfaceText("button.report_pdf", "button", "計算書PDF保存", "Save Report PDF"),
    SurfaceText("button.material_all_fields", "button", "全項目", "All Fields"),
    SurfaceText("button.material_basic_fields", "button", "基本項目", "Basic Fields"),
    SurfaceText("group.mesh_control", "form", "メッシュ制御点/ブロック分割", "Mesh Control / Blocks"),
    SurfaceText("group.element_library", "form", "GeoFEAS要素ライブラリ", "GeoFEAS Element Library"),
    SurfaceText("group.cad_layer", "form", "CADレイヤ/スナップ表示", "CAD Layers / Snap"),
    SurfaceText("group.annotation", "form", "寸法/注記", "Dimensions / Notes"),
    SurfaceText("group.material_library", "form", "GeoFEAS材料ライブラリ", "GeoFEAS Material Library"),
    SurfaceText("label.material_list", "form", "材料一覧", "Materials"),
    SurfaceText("group.solver_job", "form", "解析ジョブ", "Analysis Job"),
    SurfaceText("checkbox.axisymmetric_guide", "form", "軸対称 r-z ガイド", "Axisymmetric r-z Guide"),
    SurfaceText("group.batch_boundary", "form", "画面選択から一括境界を作成", "Create Boundary From Selection"),
    SurfaceText("group.load_cases", "form", "荷重ケース", "Load Cases"),
    SurfaceText("group.batch_load", "form", "画面選択から一括荷重を作成", "Create Loads From Selection"),
    SurfaceText("group.stage_selection", "form", "画面選択からステージ条件を作成", "Create Stage Conditions From Selection"),
    SurfaceText("group.stage_detail", "form", "選択ステージ詳細フォーム", "Selected Stage Detail"),
    SurfaceText("group.stage_diff", "form", "前ステージとの差分", "Difference From Previous Stage"),
    SurfaceText("group.output_compare", "form", "出力比較", "Output Comparison"),
    SurfaceText("group.srm_post", "form", "SRM専用Post", "SRM Post"),
    SurfaceText("group.drawing_layout", "form", "図面レイアウト", "Drawing Layout"),
    SurfaceText("group.report_wysiwyg", "form", "帳票ページWYSIWYG", "Report Page WYSIWYG"),
    SurfaceText("combo.hide_sets", "form", "セット非表示", "Hide Sets"),
    SurfaceText("combo.replace", "form", "置換", "Replace"),
    SurfaceText("combo.add", "form", "追加", "Add"),
    SurfaceText("combo.remove", "form", "解除", "Remove"),
    SurfaceText("combo.invert", "form", "反転", "Invert"),
    SurfaceText("combo.all_points", "form", "全点", "All Points"),
    SurfaceText("combo.nodes", "form", "節点", "Nodes"),
    SurfaceText("combo.geometry", "form", "形状", "Geometry"),
    SurfaceText("combo.intersections", "form", "交点", "Intersections"),
    SurfaceText("combo.grid", "form", "グリッド", "Grid"),
    SurfaceText("combo.operation_unit", "form", "操作単位", "Operation"),
    SurfaceText("combo.form_unit", "form", "フォーム単位", "Form"),
    SurfaceText("combo.manual_only", "form", "手動のみ", "Manual Only"),
    SurfaceText("combo.quality_auto", "form", "自動", "Auto"),
    SurfaceText("combo.quality_full", "form", "高品質", "Full"),
    SurfaceText("combo.quality_fast", "form", "高速", "Fast"),
    SurfaceText("combo.selected_endpoint", "form", "選択端点", "Selected Endpoint"),
    SurfaceText("combo.start_point", "form", "始点", "Start Point"),
    SurfaceText("combo.end_point", "form", "終点", "End Point"),
    SurfaceText("combo.whole_line", "form", "線全体", "Whole Line"),
    SurfaceText("combo.global_bc", "form", "全体BC", "Global BC"),
    SurfaceText("combo.selected_stage", "form", "選択ステージ", "Selected Stage"),
    SurfaceText("combo.fixed_uxuy", "form", "固定(ux=uy=0)", "Fixed (ux=uy=0)"),
    SurfaceText("combo.horizontal_roller_uy", "form", "水平ローラ(uy=0)", "Horizontal Roller (uy=0)"),
    SurfaceText("combo.vertical_roller_ux", "form", "鉛直ローラ(ux=0)", "Vertical Roller (ux=0)"),
    SurfaceText("combo.pin_uxuy", "form", "ピン(ux=uy=0)", "Pin (ux=uy=0)"),
    SurfaceText("combo.prescribed_displacement", "form", "強制変位", "Prescribed Displacement"),
    SurfaceText("combo.edge_water_pressure", "form", "辺水圧", "Edge Water Pressure"),
    SurfaceText("combo.node_water_pressure", "form", "節点水圧", "Node Water Pressure"),
    SurfaceText("combo.flow_rate", "form", "流量", "Flow Rate"),
    SurfaceText("combo.global_load", "form", "全体荷重", "Global Load"),
    SurfaceText("combo.selected_element_material", "form", "選択要素の材料", "Selected Element Material"),
    SurfaceText("combo.node_set_all", "form", "節点: all", "Nodes: all"),
    SurfaceText("combo.node_set_left", "form", "節点: left", "Nodes: left"),
    SurfaceText("combo.node_set_right", "form", "節点: right", "Nodes: right"),
    SurfaceText("combo.node_set_top", "form", "節点: top", "Nodes: top"),
    SurfaceText("combo.node_set_bottom", "form", "節点: bottom", "Nodes: bottom"),
    SurfaceText("combo.element_set_all", "form", "要素: all", "Elements: all"),
    SurfaceText("combo.uniform_distribution", "form", "等分布", "Uniform"),
    SurfaceText("combo.tapered_distribution", "form", "偏分布", "Tapered"),
    SurfaceText("combo.linear_elastic_plane_strain", "form", "線形弾性(平面ひずみ)", "Linear Elastic (Plane Strain)"),
    SurfaceText("combo.mohr_coulomb", "form", "モール・クーロン", "Mohr-Coulomb"),
    SurfaceText("combo.drucker_prager", "form", "ドラッカー・プラガー", "Drucker-Prager"),
    SurfaceText("combo.von_mises_j2", "form", "von Mises / J2塑性", "von Mises / J2 Plasticity"),
    SurfaceText("combo.tensionless_elastic", "form", "引張なし弾性", "Tensionless Elastic"),
    SurfaceText("combo.hardin_drnevich", "form", "非線形弾性: Hardin-Drnevich", "Nonlinear Elastic: Hardin-Drnevich"),
    SurfaceText("combo.duncan_chang", "form", "非線形弾性: Duncan-Chang", "Nonlinear Elastic: Duncan-Chang"),
    SurfaceText("combo.ramberg_osgood", "form", "非線形弾性: Ramberg-Osgood", "Nonlinear Elastic: Ramberg-Osgood"),
    SurfaceText("combo.liquefaction_bilinear", "form", "液状化: バイリニア", "Liquefaction: Bilinear"),
    SurfaceText("combo.liquefaction_ru_fl", "form", "液状化(ru-FL代替)", "Liquefaction (ru-FL Alt.)"),
    SurfaceText("combo.pz_clay", "form", "粘性土代替モデル(PZ)", "Clay Substitute Model (PZ)"),
    SurfaceText("combo.sand_substitute", "form", "砂質土代替モデル", "Sandy Soil Substitute Model"),
    SurfaceText("combo.clay_substitute", "form", "粘性土代替モデル", "Clay Substitute Model"),
    SurfaceText("combo.standard_report", "form", "標準帳票", "Standard Report"),
    SurfaceText("combo.geofeas_report", "form", "GeoFEAS風 判定付き", "GeoFEAS-like Review"),
    SurfaceText("combo.design_report", "form", "設計照査帳票", "Design Check Report"),
    SurfaceText("analysis.label.type", "form", "解析種別", "Analysis Type"),
    SurfaceText("analysis.label.geometry", "form", "解析形状", "Analysis Geometry"),
    SurfaceText("analysis.label.deformation_mode", "form", "変形モード", "Deformation Mode"),
    SurfaceText("analysis.label.coupling", "form", "連成", "Coupling"),
    SurfaceText("analysis.label.unit_system", "form", "単位系", "Unit System"),
    SurfaceText("analysis.checkbox.up", "button", "u-p/圧密フィールド", "u-p / Consolidation Fields"),
    SurfaceText("analysis.group.input_template", "form", "入力テンプレート", "Input Template"),
    SurfaceText("analysis.button.load_selected_template", "button", "選択テンプレートを読込", "Load Selected Template"),
    SurfaceText("analysis.button.load_template_file", "button", "ファイルから読込", "Load From File"),
    SurfaceText("analysis.tooltip.input_template", "form", "標準サンプル、examples、プロジェクトtemplates配下の入力テンプレートを選びます。", "Choose an input template from the standard sample, examples, or project templates folder."),
    SurfaceText("analysis.template.standard_sample", "form", "標準2Dサンプル", "Standard 2D Sample"),
    SurfaceText("analysis.group.axisymmetric", "form", "軸対称プリセット", "Axisymmetric Presets"),
    SurfaceText("analysis.button.reference_sets", "button", "r/z集合", "r/z Sets"),
    SurfaceText("analysis.button.standard_settings", "button", "標準設定", "Standard Settings"),
    SurfaceText("analysis.group.output_location", "form", "出力先", "Output Location"),
    SurfaceText("analysis.label.output_policy", "form", "場所", "Location"),
    SurfaceText("analysis.label.output_custom", "form", "任意", "Custom"),
    SurfaceText("analysis.label.output_formats", "form", "形式", "Formats"),
    SurfaceText("analysis.label.output_resolved", "form", "解決後", "Resolved"),
    SurfaceText("analysis.output.same_as_input", "form", "入力YAMLと同じ場所", "Same as Input YAML"),
    SurfaceText("analysis.output.project_runs", "form", "プロジェクト runs", "Project runs"),
    SurfaceText("analysis.output.custom_folder", "form", "任意フォルダー", "Custom Folder"),
    SurfaceText("analysis.button.output_browse", "button", "参照", "Browse"),
    SurfaceText("analysis.button.apply", "button", "解析条件を反映", "Apply Analysis Settings"),
    SurfaceText("analysis.summary.2d", "form", "2D専用。平面ひずみ/軸対称を切替可能。3D解析経路はありません。", "2D only. Plane strain and axisymmetric modes are available; there is no 3D analysis path."),
    SurfaceText("tooltip.select_by_condition", "form", "条件式で選択します。", "Select by condition expression."),
    SurfaceText("tooltip.draw_closed_region", "form", "閉領域を作図します。", "Draw a closed region."),
    SurfaceText("tooltip.undo_edit", "form", "直前の編集を元に戻します。", "Undo the last edit."),
    SurfaceText("tooltip.redo_edit", "form", "元に戻した編集をやり直します。", "Redo the undone edit."),
    SurfaceText("tooltip.cad_undo_edit", "form", "直前の形状/CAD編集を元に戻します。", "Undo the last Geometry/CAD edit."),
    SurfaceText("tooltip.cad_redo_edit", "form", "Undoで戻した形状/CAD編集をやり直します。", "Redo the undone Geometry/CAD edit."),
    SurfaceText("tooltip.undo_history", "form", "直前の編集を元に戻します。履歴は最大10件です。", "Undo the last edit. History stores up to 10 entries."),
    SurfaceText("tooltip.redo_history", "form", "Undoで戻した編集をやり直します。履歴は最大10件です。", "Redo the edit restored by Undo. History stores up to 10 entries."),
    SurfaceText("tooltip.cad_select", "form", "通常選択に戻します。表示中の線分・端点を選択できます。", "Return to normal selection. Select visible lines and endpoints."),
    SurfaceText("tooltip.cad_part_select", "form", "端点や制御点を選択して座標微修正します。", "Select endpoints or control points for coordinate adjustments."),
    SurfaceText("tooltip.model_check_precheck", "form", "入力とモデルを事前チェックします。", "Pre-check the input and model."),
    SurfaceText("tooltip.finish_region", "form", "作図中の領域を閉じて確定します。", "Close and confirm the region being drawn."),
    SurfaceText("tooltip.rectangle_select_range", "form", "矩形範囲で選択します。", "Select by a rectangular range."),
    SurfaceText("tooltip.lasso_select_range", "form", "投げ縄範囲で選択します。", "Select by a lasso range."),
    SurfaceText("tooltip.filter_select_targets", "form", "条件で対象を選択します。", "Select targets by condition."),
    SurfaceText("tooltip.clear_current_selection", "form", "現在の選択を解除します。", "Clear the current selection."),
    SurfaceText("tooltip.invert_current_selection", "form", "選択状態を反転します。", "Invert the selection state."),
    SurfaceText("tooltip.record_current_selection", "form", "現在の選択を履歴へ記録します。", "Record the current selection to history."),
    SurfaceText("tooltip.save_named_current_selection", "form", "現在の選択に名前を付けて保存します。", "Save the current selection with a name."),
    SurfaceText("tooltip.compare_saved_selections", "form", "保存済み選択セットを比較します。", "Compare saved selection sets."),
    SurfaceText("tooltip.selection_mode_help", "form", "選択モードの説明を表示します。", "Show selection mode help."),
    SurfaceText("tooltip.save_current_selection_set", "form", "現在の選択をsetへ登録します。", "Register the current selection as a set."),
    SurfaceText("tooltip.cad_line", "form", "2点クリックでメッシュ分割対象の直線を作成します。", "Create a mesh-splitting line with two clicks."),
    SurfaceText("tooltip.cad_polyline", "form", "連続クリックで折れ線を作成し、右クリック/Enterで確定します。", "Create a polyline with repeated clicks; right-click or press Enter to finish."),
    SurfaceText("tooltip.cad_rectangle", "form", "対角2点クリックで矩形領域を作図します。", "Draw a rectangular region with two diagonal corner clicks."),
    SurfaceText("tooltip.cad_polygon", "form", "複数点クリックで閉合領域を作図します。", "Draw a closed polygonal region with multiple clicks."),
    SurfaceText("tooltip.cad_circle", "form", "中心点と半径点をクリックして円形境界を作図します。", "Draw a circular boundary by clicking the center and radius point."),
    SurfaceText("tooltip.cad_arc", "form", "3点クリックで円弧を作図します。", "Draw an arc with three clicks."),
    SurfaceText("tooltip.cad_curve", "form", "制御点を連続クリックし、右クリック/Enterで曲線近似線を確定します。", "Click control points, then right-click or press Enter to create an approximated curve."),
    SurfaceText("tooltip.cad_point", "form", "ID点/補助点を作図します。", "Draw an ID point or helper point."),
    SurfaceText("tooltip.cad_eraser", "form", "モデルビューで選択した線・領域・トンネルを削除します。", "Delete selected lines, regions, or tunnels in the model view."),
    SurfaceText("tooltip.cad_move", "form", "端点ドラッグまたは座標微修正で移動します。", "Move by endpoint dragging or coordinate adjustment."),
    SurfaceText("tooltip.cad_trim", "form", "選択線を指定範囲でトリムします。", "Trim selected lines by the specified range."),
    SurfaceText("tooltip.cad_extend", "form", "選択線を延長します。", "Extend the selected line."),
    SurfaceText("tooltip.cad_split", "form", "交点で線分を分割します。", "Split lines at intersections."),
    SurfaceText("tooltip.cad_join", "form", "選択端点を近接端点へスナップして結合します。", "Snap and join the selected endpoint to a nearby endpoint."),
    SurfaceText("tooltip.cad_close", "form", "選択端点から近接する未閉合端点へ閉合線を追加します。", "Add a closing segment from the selected endpoint to a nearby open endpoint."),
    SurfaceText("tooltip.cad_region", "form", "選択した閉合線分を形状領域として登録します。", "Register selected closed line segments as a geometry region."),
    SurfaceText("tooltip.cad_grid_condition", "form", "選択した形状領域に要素種別と格子密度を設定します。形状/CADではメッシュを表示しません。", "Set element type and grid density for selected geometry regions. Geometry/CAD does not display the mesh."),
    SurfaceText("tooltip.cad_helper_line", "form", "2点クリックでメッシュ分割対象外の補助線を作成します。", "Create a helper line excluded from mesh splitting with two clicks."),
    SurfaceText("tooltip.cad_helper_convert", "form", "選択線分をメッシュ分割対象外の補助線へ変更します。", "Convert selected lines to helper lines excluded from mesh splitting."),
    SurfaceText("tooltip.cad_model_convert", "form", "選択線分をメッシュ分割対象へ戻します。", "Return selected lines to mesh-splitting model lines."),
    SurfaceText("tooltip.grid_toggle", "form", "X/Yグリッド線の表示/非表示を切り替えます。", "Toggle X/Y grid line visibility."),
    SurfaceText("tooltip.fit_current_model", "form", "現在のモデル全体を表示します。", "Fit the current model to the view."),
    SurfaceText("tooltip.mesh_select", "form", "メッシュ制御とメッシュ要素を選択します。", "Select mesh controls and mesh elements."),
    SurfaceText("tooltip.mesh_undo", "form", "直前のメッシュ編集または形状編集を戻します。", "Undo the last mesh or geometry edit."),
    SurfaceText("tooltip.mesh_redo", "form", "Undoで戻した編集をやり直します。", "Redo the edit restored by Undo."),
    SurfaceText("tooltip.mesh_control_point", "form", "選択点または形状中心へメッシュ制御点を追加します。", "Add a mesh control point at the selection or geometry center."),
    SurfaceText("tooltip.mesh_refine", "form", "選択点または形状中心へ局所細分範囲を追加します。", "Add a local refinement range at the selection or geometry center."),
    SurfaceText("tooltip.mesh_split_line", "form", "選択した2点/辺、または形状中心線をメッシュ分割線にします。", "Create a mesh split line from two selected points/edges or a geometry centerline."),
    SurfaceText("tooltip.mesh_size", "form", "選択点まわりへ局所メッシュサイズを設定します。", "Set a local mesh size around the selected point."),
    SurfaceText("tooltip.mesh_block_split", "form", "選択要素範囲にブロック分割ヒントを追加します。", "Add a block-splitting hint to the selected element range."),
    SurfaceText("tooltip.mesh_quality_extract", "form", "メッシュ品質違反を抽出して表示します。", "Extract and display mesh quality violations."),
    SurfaceText("tooltip.mesh_violation_select", "form", "品質違反要素をモデルビューで選択します。", "Select quality-violation elements in the model view."),
    SurfaceText("tooltip.mesh_violation_repair", "form", "選択した品質違反要素へ局所細分と点群再配置用の制御点を追加します。", "Add local refinement and point-redistribution controls for selected quality violations."),
    SurfaceText("tooltip.mesh_delete_control", "form", "選択中のメッシュ制御を削除します。要素選択時はその形状のメッシュだけを削除します。", "Delete selected mesh controls. When elements are selected, delete only that geometry's mesh."),
    SurfaceText("tooltip.mesh_delete", "form", "現在のメッシュ結果を削除し、再構成待ちにします。", "Delete the current mesh result and mark it for rebuild."),
    SurfaceText("tooltip.mesh_rebuild", "form", "形状とメッシュ制御からメッシュを作り直します。", "Rebuild the mesh from geometry and mesh controls."),
    SurfaceText("tooltip.mesh_apply_controls", "form", "右パネルのメッシュ制御表を設定へ反映します。", "Apply the right-panel mesh control table to settings."),
    SurfaceText("tooltip.mesh_fit", "form", "現在の形状とメッシュ範囲を表示します。", "Fit the current geometry and mesh range to the view."),
    SurfaceText("tooltip.disabled", "form", "無効", "Disabled"),
    SurfaceText("tooltip.disabled.nodes_or_edges", "form", "節点または辺を選択してください。", "Select nodes or edges."),
    SurfaceText("tooltip.disabled.hydraulic_edges", "form", "水理境界を設定する辺または要素境界を選択してください。", "Select edges or element boundaries for hydraulic boundaries."),
    SurfaceText("tooltip.disabled.mpc_nodes", "form", "MPCには節点を2つ以上選択してください。", "Select at least two nodes for MPC."),
    SurfaceText("tooltip.disabled.nodal_load_nodes", "form", "節点荷重を設定する節点を選択してください。", "Select nodes for nodal loads."),
    SurfaceText("tooltip.disabled.body_force_target", "form", "体積力は要素を選択するか、材料名を指定してください。", "Select elements or specify a material name for body force."),
    SurfaceText("tooltip.disabled.surface_load_edges", "form", "面荷重を設定する辺または要素境界を選択してください。", "Select edges or element boundaries for surface loads."),
    SurfaceText("placeholder.cad_input", "form", "CAD入力: line 0,0,1,0 / circle 0,0,1", "CAD Input: line 0,0,1,0 / circle 0,0,1"),
    SurfaceText("placeholder.prescribed_ux", "form", "強制変位ux。空欄なら未指定", "Prescribed ux. Leave blank if unspecified."),
    SurfaceText("placeholder.prescribed_uy", "form", "強制変位uy。空欄なら未指定", "Prescribed uy. Leave blank if unspecified."),
    SurfaceText("placeholder.first_selected_node", "form", "空欄なら選択節点の先頭", "Blank uses the first selected node."),
    SurfaceText("geometry.label.target", "form", "対象", "Target"),
    SurfaceText("geometry.label.edit_target", "form", "編集対象", "Edit Target"),
    SurfaceText("geometry.label.fine_adjust", "form", "微修正", "Fine Adjust"),
    SurfaceText("geometry.label.length", "form", "長さ", "Length"),
    SurfaceText("geometry.label.angle", "form", "角度", "Angle"),
    SurfaceText("geometry.label.cad_input", "form", "CAD入力", "CAD Input"),
    SurfaceText("mesh.label.element", "form", "要素", "Element"),
    SurfaceText("mesh.label.density_size", "form", "密度/サイズ", "Density/Size"),
    SurfaceText("mesh.label.manual_control", "form", "手動制御", "Manual Control"),
    SurfaceText("mesh.label.radius", "form", "半径", "Radius"),
    SurfaceText("mesh.label.gradient", "form", "勾配", "Gradient"),
    SurfaceText("mode.current_select", "form", "現在: 選択", "Current: Select"),
    SurfaceText("mode.current_mesh_manual", "form", "現在: メッシュ手動編集（節点ドラッグ/形状別条件）", "Current: Manual Mesh Edit (node drag / per-geometry settings)"),
    SurfaceText("disabled.no_selection", "form", "選択対象がありません。", "No selection is available."),
    SurfaceText("disabled.cad_region_points", "form", "領域作図で3点以上入力すると有効です。", "Available after entering at least three region-drawing points."),
    SurfaceText("disabled.geometry_shape_selection", "form", "形状を選択してください。", "Select a geometry item."),
    SurfaceText("disabled.geometry_line_selection", "form", "線分または端点を選択してください。", "Select a line segment or endpoint."),
    SurfaceText("disabled.geometry_endpoint_selection", "form", "端点を選択してください。", "Select an endpoint."),
    SurfaceText("disabled.geometry_region_selection", "form", "形状領域を選択してください。", "Select a geometry region."),
    SurfaceText("disabled.undo_history", "form", "Undoできる履歴がありません。", "No undo history is available."),
    SurfaceText("disabled.redo_history", "form", "Redoできる履歴がありません。", "No redo history is available."),
    SurfaceText("disabled.node_or_element_selection", "form", "節点または要素を選択してください。", "Select nodes or elements."),
    SurfaceText("disabled.element_selection", "form", "要素を選択してください。", "Select elements."),
    SurfaceText("disabled.mesh_control_or_element_selection", "form", "メッシュ制御または要素を選択してください。", "Select mesh controls or elements."),
    SurfaceText("disabled.mesh_quality_rows", "form", "品質違反を抽出してください。", "Extract quality violations first."),
    SurfaceText("disabled.mesh_quality_selection", "form", "品質違反要素を選択してください。", "Select quality-violation elements."),
    SurfaceText("disabled.named_selection_pair", "form", "比較する名前付き選択が2件以上必要です。", "At least two named selections are required for comparison."),
    SurfaceText("boundary.description.table", "form", "境界条件を表で編集します。複雑なMPC等は下のYAML詳細で保持・編集できます。", "Edit boundary conditions in the table. Complex MPC data is kept in the YAML detail."),
    SurfaceText("load.description.table", "form", "荷重を節点/辺/自重の表で編集します。高度な荷重は下のYAML詳細で保持・編集できます。", "Edit loads in the node, edge, and self-weight tables. Advanced loads are kept in the YAML detail."),
    SurfaceText("selection.description.model_view", "form", "モデルビューで節点・辺・要素を複数選択してから操作します。", "Select nodes, edges, or elements in the model view before applying actions."),
    SurfaceText("condition.label.target_scope", "form", "反映先", "Target Scope"),
    SurfaceText("boundary.label.support", "form", "支点/変位", "Support/Displacement"),
    SurfaceText("boundary.label.hydraulic", "form", "水理条件", "Hydraulic Condition"),
    SurfaceText("boundary.label.hydraulic_value", "form", "水理値", "Hydraulic Value"),
    SurfaceText("load.label.body_force_material", "form", "体積力材料", "Body-force Material"),
    SurfaceText("load.label.surface_distribution", "form", "面荷重分布", "Surface-load Distribution"),
    SurfaceText("load.label.body_force_bx", "form", "体積力 bx", "Body Force bx"),
    SurfaceText("load.label.body_force_by", "form", "体積力 by", "Body Force by"),
    SurfaceText("load.label.nodal_fx", "form", "節点 fx", "Node fx"),
    SurfaceText("load.label.nodal_fy", "form", "節点 fy", "Node fy"),
    SurfaceText("load.label.distributed_tx", "form", "分布 tx", "Distributed tx"),
    SurfaceText("load.label.distributed_ty", "form", "分布 ty", "Distributed ty"),
    SurfaceText("load.label.end_tx", "form", "終端 tx", "End tx"),
    SurfaceText("load.label.end_ty", "form", "終端 ty", "End ty"),
    SurfaceText("load.label.seismic_kh", "form", "水平震度 kh", "Seismic kh"),
    SurfaceText("load.label.seismic_kv", "form", "鉛直震度 kv", "Seismic kv"),
    SurfaceText("load.label.seismic_direction", "form", "地震方向", "Seismic Direction"),
    SurfaceText("stage.label.stage_list", "form", "ステージ一覧", "Stage List"),
    SurfaceText("stage.label.change_input", "form", "ステージごとの変更入力（材料・境界・荷重・無効化/再有効化）", "Stage-by-stage Changes (Materials, Boundaries, Loads, Deactivate/Reactivate)"),
    SurfaceText("model_check.label.yaml_state", "form", "YAML入力状態", "YAML Input State"),
    SurfaceText("model_check.label.check_range", "form", "1解析から7ステージまでの確認状況", "Check Status from 1 Analysis through 7 Stages"),
    SurfaceText("model_check.label.results", "form", "モデルチェック結果", "Model Check Results"),
    SurfaceText("solver.label.state_idle", "form", "解析実行状態: 待機中", "Run State: Idle"),
    SurfaceText("solver.label.state_running", "form", "解析実行状態: 実行中", "Run State: Running"),
    SurfaceText("solver.label.log", "form", "解析実行ログ", "Run Log"),
    SurfaceText("solver.group.final_summary", "form", "解析実行前の最終サマリ", "Final Pre-run Summary"),
    SurfaceText("table.area", "form", "領域", "Area"),
    SurfaceText("table.item", "form", "項目", "Item"),
    SurfaceText("table.unit", "form", "単位", "Unit"),
    SurfaceText("table.recommended", "form", "推奨", "Recommended"),
    SurfaceText("table.prohibited", "form", "禁止", "Forbidden"),
    SurfaceText("table.status", "form", "状態", "Status"),
    SurfaceText("table.content", "form", "内容", "Content"),
    SurfaceText("table.template", "form", "テンプレート", "Template"),
    SurfaceText("table.operation", "form", "操作", "Operation"),
    SurfaceText("table.nodes", "form", "節点", "Nodes"),
    SurfaceText("table.edges", "form", "辺", "Edges"),
    SurfaceText("table.elements", "form", "要素", "Elements"),
    SurfaceText("table.severity", "form", "区分", "Category"),
    SurfaceText("table.target", "form", "対象", "Target"),
    SurfaceText("table.message", "form", "内容", "Message"),
    SurfaceText("table.workflow_step", "form", "工程", "Step"),
    SurfaceText("table.missing_check", "form", "不足/確認", "Missing/Check"),
    SurfaceText("table.next_action", "form", "次の操作", "Next Action"),
    SurfaceText("table.material_name", "form", "材料名", "Material Name"),
    SurfaceText("table.young_modulus", "form", "E(ヤング率)", "E (Young's Modulus)"),
    SurfaceText("table.poisson_ratio", "form", "ν(ポアソン比)", "nu (Poisson's Ratio)"),
    SurfaceText("table.unit_weight", "form", "γ(単位体積重量)", "gamma (Unit Weight)"),
    SurfaceText("table.cohesion", "form", "粘着力 c", "Cohesion c"),
    SurfaceText("table.friction_angle", "form", "内部摩擦角 φ", "Friction Angle phi"),
    SurfaceText("table.dilatancy_angle", "form", "ダイレイタンシー角 ψ", "Dilatancy Angle psi"),
    SurfaceText("table.yield_stress", "form", "降伏応力", "Yield Stress"),
    SurfaceText("table.hardening", "form", "硬化係数", "Hardening"),
    SurfaceText("table.tension_cutoff", "form", "引張カット", "Tension Cutoff"),
    SurfaceText("table.tensile_strength", "form", "引張強度 ft", "Tensile Strength ft"),
    SurfaceText("table.extra_yaml", "form", "追加YAML", "Extra YAML"),
    SurfaceText("table.target_node_set", "form", "対象(節点/セット)", "Target (Node/Set)"),
    SurfaceText("table.case", "form", "ケース", "Case"),
    SurfaceText("table.type", "form", "種別", "Type"),
    SurfaceText("table.scale", "form", "倍率", "Scale"),
    SurfaceText("table.active", "form", "有効", "Active"),
    SurfaceText("table.description", "form", "説明", "Description"),
    SurfaceText("table.target_node_set_edge", "form", "対象(節点/セット/辺)", "Target (Node/Set/Edge)"),
    SurfaceText("table.action_condition", "form", "action/条件", "Action/Condition"),
    SurfaceText("table.stress_release", "form", "応力解放", "Stress Release"),
    SurfaceText("table.check_item", "form", "確認項目", "Check Item"),
    SurfaceText("table.value", "form", "値", "Value"),
    SurfaceText("table.result_status", "form", "状態", "Status"),
    SurfaceText("cell.input_assistance", "form", "入力補助", "Input Assistance"),
    SurfaceText("cell.material_count", "form", "材料数", "Materials"),
    SurfaceText("cell.unassigned", "form", "未割当", "Unassigned"),
    SurfaceText("cell.boundary_count", "form", "境界条件数", "Boundary Conditions"),
    SurfaceText("cell.load_count", "form", "荷重条件数", "Loads"),
    SurfaceText("cell.stage_count", "form", "ステージ数", "Stages"),
    SurfaceText("cell.mesh_quality", "form", "メッシュ品質", "Mesh Quality"),
    SurfaceText("cell.check", "form", "確認", "Check"),
    SurfaceText("status.project.locked", "status", "プロジェクトをロックしました: {owner}", "Project locked: {owner}"),
    SurfaceText("status.project.unlocked", "status", "プロジェクトロックを解除しました。", "Project lock released."),
    SurfaceText("status.autosave.running", "status", "自動保存を実行中です。", "Autosave is running."),
    SurfaceText("status.autosave.no_change", "status", "自動保存: 変更なし", "Autosave: no changes"),
    SurfaceText("status.autosave.background", "status", "自動保存をバックグラウンド実行中...", "Autosave is running in the background..."),
    SurfaceText("status.workspace.updated", "status", "ワークスペース概要を更新しました: {path}", "Workspace dashboard updated: {path}"),
    SurfaceText("status.customization.applied", "status", "組織プロファイルを適用しました: {path}", "Organization profile applied: {path}"),
    SurfaceText("status.model_check.running", "status", "モデルチェックを実行中です。", "Model check is running."),
    SurfaceText("status.preflight.running", "status", "解析直前モデルチェックを実行中です。", "Pre-run model check is running."),
    SurfaceText("status.model_check.summary", "status", "エラー {errors} / 警告 {warnings} / 情報 {infos}", "ERROR {errors} / WARN {warnings} / INFO {infos}"),
    SurfaceText("status.action.new_sample", "status", "新しい2Dサンプル入力を作成します。", "Create a new 2D sample input."),
    SurfaceText("status.action.open_input", "status", "YAML入力ファイルを開きます。", "Open a YAML input file."),
    SurfaceText("status.action.save_input", "status", "現在の入力を保存します。", "Save the current input."),
    SurfaceText("status.action.save_as", "status", "保存先を指定して入力を保存します。", "Choose a destination and save the input."),
    SurfaceText("status.action.import_project_template", "status", "形状、材料、境界、荷重、ステージを含む入力テンプレートを読込みます。", "Load an input template including geometry, materials, boundaries, loads, and stages."),
    SurfaceText("status.no_results", "result", "まだ解析結果がありません。", "No analysis results yet."),
    SurfaceText(
        "status.no_results.result_navigation",
        "result",
        "解析結果がまだありません。解析実行後に結果図へ移動できます。",
        "No analysis results are available. Run the analysis before opening result views.",
    ),
    SurfaceText(
        "status.no_results.report_navigation",
        "report",
        "解析結果がまだありません。解析実行後に帳票へ移動できます。",
        "No analysis results are available. Run the analysis before opening reports.",
    ),
    SurfaceText(
        "status.no_results.guidance",
        "result",
        "解析結果がまだありません。入力課題を解消して解析を実行してください。",
        "No analysis results are available. Resolve input issues and run the analysis.",
    ),
    SurfaceText("status.report.note", "report", "解析後のresults配下にHTMLレポートを生成します。現在は結果フォルダとsummaryを確認してください。", "HTML reports are generated under results after analysis. Check the result folder and summary."),
    SurfaceText("log.yaml.applied", "log", "[GUI] YAMLをフォームへ反映しました", "[GUI] Applied YAML to forms"),
    SurfaceText("log.sample.created", "log", "[GUI] 2Dサンプルを作成しました", "[GUI] Created 2D sample"),
    SurfaceText("log.workspace.updated", "log", "[GUI] ワークスペース概要を更新しました: {path}", "[GUI] Workspace dashboard updated: {path}"),
    SurfaceText("log.customization.applied", "log", "[GUI] 組織プロファイルを適用しました: {path}", "[GUI] Organization profile applied: {path}"),
    SurfaceText("log.model_check.done", "log", "[GUI] モデルチェック完了: {count}件", "[GUI] Model check completed: {count} items"),
    SurfaceText("log.report.selected", "log", "[GeoFEAS操作] 計算書作成を選択しました", "[GeoFEAS Workflow] Selected report builder"),
    SurfaceText("log.version.shown", "log", "[GUI] バージョン情報を表示しました", "[GUI] Version information shown"),
    SurfaceText("dialog.version.title", "dialog", "バージョン情報", "Version Information"),
)


def surface_text_catalog(*, locale: str = DEFAULT_GUI_LOCALE) -> list[dict[str, str]]:
    return [
        {
            "key": row.key,
            "category": row.category,
            "ja": row.ja,
            "en": row.en,
            "text": _text_for_locale(row, locale),
        }
        for row in SURFACE_TEXTS
    ]


def gui_surface_message(message_key: str, *, locale: str = DEFAULT_GUI_LOCALE, **values: Any) -> str:
    row = next((item for item in SURFACE_TEXTS if item.key == message_key), None)
    text = _text_for_locale(row, locale) if row else message_key
    if values:
        try:
            return text.format(**values)
        except Exception:
            return text
    return text


@lru_cache(maxsize=None)
def _surface_text_lookup_maps(locale: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    all_texts: dict[str, str] = {}
    by_category: dict[str, dict[str, str]] = {}
    for row in SURFACE_TEXTS:
        text = _text_for_locale(row, locale)
        all_texts.setdefault(row.ja, text)
        all_texts.setdefault(row.en, text)
        category_map = by_category.setdefault(row.category, {})
        category_map.setdefault(row.ja, text)
        category_map.setdefault(row.en, text)
    return all_texts, by_category


@lru_cache(maxsize=8192)
def translate_surface_tooltip(text: str, *, locale: str = DEFAULT_GUI_LOCALE) -> str:
    return _translate_rich_tooltip(str(text or ""), locale=locale)


@lru_cache(maxsize=16384)
def translate_surface_text(text: str, *, locale: str = DEFAULT_GUI_LOCALE, category: str | None = None) -> str:
    normalized = str(text)
    all_texts, by_category = _surface_text_lookup_maps(locale)
    if category:
        translated = by_category.get(category, {}).get(normalized)
        if translated is not None:
            return translated
    translated = all_texts.get(normalized)
    if translated is not None:
        return translated
    dynamic = _translate_dynamic_surface_text(normalized, locale=locale)
    if dynamic != normalized:
        return dynamic
    return normalized


@lru_cache(maxsize=8192)
def _translate_dynamic_surface_text(text: str, *, locale: str = DEFAULT_GUI_LOCALE) -> str:
    if not locale.startswith("en"):
        return text
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
    match = re.fullmatch(r"数値表: (.*)", text)
    if match:
        detail = re.sub(r"(\d+)件", r"\1 rows", match.group(1))
        return f"Value Table: {detail}"
    return text


@lru_cache(maxsize=8192)
def _translate_rich_tooltip(text: str, *, locale: str = DEFAULT_GUI_LOCALE) -> str:
    raw = str(text or "")
    if not raw:
        return raw
    direct = translate_surface_text(raw, locale=locale)
    if direct != raw:
        return direct
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.startswith("Help:") or stripped.startswith(("http://", "https://")):
            lines.append(line)
            continue
        translated = _translate_tooltip_line(line, locale=locale)
        if locale.startswith("en") and _CJK_RE.search(translated):
            # Keep the help link but do not show stale Japanese in English mode.
            continue
        lines.append(translated)
    return "\n".join(lines).strip()


@lru_cache(maxsize=8192)
def _translate_tooltip_line(line: str, *, locale: str = DEFAULT_GUI_LOCALE) -> str:
    text = str(line)
    direct = translate_surface_text(text, locale=locale)
    if direct != text:
        return direct
    match = re.fullmatch(r"<b>(.*?)</b>(?:<br>(.*))?", text, flags=re.DOTALL)
    if match:
        title = translate_surface_text(match.group(1), locale=locale, category="button")
        body_raw = match.group(2) or ""
        body = _translate_tooltip_line(body_raw, locale=locale) if body_raw else ""
        if locale.startswith("en") and _CJK_RE.search(body):
            body = ""
        return f"<b>{title}</b>" + (f"<br>{body}" if body else "")
    separator = "：" if "：" in text else (":" if ":" in text else "")
    if separator:
        prefix, suffix = text.split(separator, 1)
        translated_prefix = translate_surface_text(prefix.strip(), locale=locale)
        translated_suffix = translate_surface_text(suffix.strip(), locale=locale)
        if translated_prefix != prefix.strip() or translated_suffix != suffix.strip():
            return f"{translated_prefix}: {translated_suffix}" if translated_suffix else translated_prefix
    return text


def validate_surface_text_catalog() -> dict[str, Any]:
    errors: list[str] = []
    keys = [row.key for row in SURFACE_TEXTS]
    if len(keys) != len(set(keys)):
        errors.append("surface text key is duplicated")
    categories = {row.category for row in SURFACE_TEXTS}
    missing_categories = REQUIRED_SURFACE_CATEGORIES - categories
    if missing_categories:
        errors.append("missing categories: " + ", ".join(sorted(missing_categories)))
    for row in SURFACE_TEXTS:
        for locale in SUPPORTED_GUI_LOCALES:
            if not _text_for_locale(row, locale):
                errors.append(f"{row.key} has empty {locale} text")
    return {
        "schema": GUI_SURFACE_TEXT_VALIDATION_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "text_count": len(SURFACE_TEXTS),
        "categories": sorted(categories),
        "locales": list(SUPPORTED_GUI_LOCALES),
    }


def apply_gui_surface_texts(window: Any, *, locale: str = DEFAULT_GUI_LOCALE, qt: Mapping[str, Any]) -> dict[str, Any]:
    translated = 0
    class_categories = {"QAbstractButton": "button"}
    for class_name in ("QAbstractButton", "QLabel"):
        cls = qt.get(class_name)
        if cls is None:
            continue
        for widget in window.findChildren(cls):
            if hasattr(widget, "text") and hasattr(widget, "setText"):
                old = str(widget.text())
                new = translate_surface_text(old, locale=locale, category=class_categories.get(class_name))
                if new != old:
                    widget.setText(new)
                    translated += 1
            if hasattr(widget, "toolTip") and hasattr(widget, "setToolTip"):
                old_tip = str(widget.toolTip())
                new_tip = _translate_rich_tooltip(old_tip, locale=locale)
                if new_tip != old_tip:
                    widget.setToolTip(new_tip)
                    translated += 1
    group_cls = qt.get("QGroupBox")
    if group_cls is not None:
        for group in window.findChildren(group_cls):
            old = str(group.title())
            new = translate_surface_text(old, locale=locale, category="form")
            if new != old:
                group.setTitle(new)
                translated += 1
            if hasattr(group, "toolTip") and hasattr(group, "setToolTip"):
                old_tip = str(group.toolTip())
                new_tip = _translate_rich_tooltip(old_tip, locale=locale)
                if new_tip != old_tip:
                    group.setToolTip(new_tip)
                    translated += 1
    tab_cls = qt.get("QTabWidget")
    if tab_cls is not None:
        for tabs in window.findChildren(tab_cls):
            for index in range(tabs.count()):
                old = str(tabs.tabText(index))
                new = translate_surface_text(old, locale=locale, category="tab")
                if new != old:
                    tabs.setTabText(index, new)
                    translated += 1
    combo_cls = qt.get("QComboBox")
    if combo_cls is not None:
        for combo in window.findChildren(combo_cls):
            if hasattr(combo, "toolTip") and hasattr(combo, "setToolTip"):
                old_tip = str(combo.toolTip())
                new_tip = _translate_rich_tooltip(old_tip, locale=locale)
                if new_tip != old_tip:
                    combo.setToolTip(new_tip)
                    translated += 1
            line_edit = combo.lineEdit() if hasattr(combo, "lineEdit") else None
            if line_edit is not None and hasattr(line_edit, "placeholderText"):
                old_placeholder = str(line_edit.placeholderText())
                new_placeholder = translate_surface_text(old_placeholder, locale=locale, category="form")
                if new_placeholder != old_placeholder:
                    line_edit.setPlaceholderText(new_placeholder)
                    translated += 1
            for index in range(combo.count()):
                old = str(combo.itemText(index))
                new = translate_surface_text(old, locale=locale, category="form")
                if new != old:
                    combo.setItemText(index, new)
                    translated += 1
    line_edit_cls = qt.get("QLineEdit")
    if line_edit_cls is not None:
        for line_edit in window.findChildren(line_edit_cls):
            old_placeholder = str(line_edit.placeholderText())
            new_placeholder = translate_surface_text(old_placeholder, locale=locale, category="form")
            if new_placeholder != old_placeholder:
                line_edit.setPlaceholderText(new_placeholder)
                translated += 1
            old_tip = str(line_edit.toolTip()) if hasattr(line_edit, "toolTip") else ""
            new_tip = _translate_rich_tooltip(old_tip, locale=locale)
            if hasattr(line_edit, "setToolTip") and new_tip != old_tip:
                line_edit.setToolTip(new_tip)
                translated += 1
    table_cls = qt.get("QTableWidget")
    if table_cls is not None:
        for table in window.findChildren(table_cls):
            for index in range(table.columnCount()):
                item = table.horizontalHeaderItem(index)
                if item is None:
                    continue
                old = str(item.text())
                new = translate_surface_text(old, locale=locale, category="form")
                if new != old:
                    item.setText(new)
                    translated += 1
            for index in range(table.rowCount()):
                item = table.verticalHeaderItem(index)
                if item is None:
                    continue
                old = str(item.text())
                new = translate_surface_text(old, locale=locale, category="form")
                if new != old:
                    item.setText(new)
                    translated += 1
            for row in range(table.rowCount()):
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    if item is None:
                        continue
                    old = str(item.text())
                    new = translate_surface_text(old, locale=locale, category="form")
                    if new != old:
                        item.setText(new)
                        translated += 1
                    old_tip = str(item.toolTip())
                    new_tip = _translate_rich_tooltip(old_tip, locale=locale)
                    if new_tip != old_tip:
                        item.setToolTip(new_tip)
                        translated += 1
    action_cls = qt.get("QAction")
    if action_cls is not None:
        for action in window.findChildren(action_cls):
            old = str(action.text()).replace("&", "")
            action_menu = action.menu() if hasattr(action, "menu") else None
            new = translate_surface_text(old, locale=locale, category="menu" if action_menu is not None else "button")
            if new != old:
                action.setText(new)
                translated += 1
            if hasattr(action, "toolTip") and hasattr(action, "setToolTip"):
                old_tip = str(action.toolTip())
                new_tip = _translate_rich_tooltip(old_tip, locale=locale)
                if new_tip != old_tip:
                    action.setToolTip(new_tip)
                    translated += 1
            if hasattr(action, "statusTip") and hasattr(action, "setStatusTip"):
                old_tip = str(action.statusTip())
                new_tip = _translate_rich_tooltip(old_tip, locale=locale)
                if new_tip != old_tip:
                    action.setStatusTip(new_tip)
                    translated += 1
    menu_cls = qt.get("QMenu")
    if menu_cls is not None:
        for menu in window.findChildren(menu_cls):
            old = str(menu.title()).replace("&", "")
            new = translate_surface_text(old, locale=locale, category="menu")
            if new != old:
                menu.setTitle(new)
                translated += 1
    tree_cls = qt.get("QTreeWidget")
    if tree_cls is not None:
        for tree in window.findChildren(tree_cls):
            old_header = str(tree.headerItem().text(0)) if tree.headerItem() is not None else ""
            new_header = translate_surface_text(old_header, locale=locale, category="tree")
            if tree.headerItem() is not None and new_header != old_header:
                tree.headerItem().setText(0, new_header)
                translated += 1
            for index in range(tree.topLevelItemCount()):
                translated += _translate_tree_item(tree.topLevelItem(index), locale)
    return {"schema": "geofem.gui.surface_text_application.v1", "locale": locale, "translated_count": translated}


def write_surface_text_catalog(output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "gui_surface_text_catalog.json"
    csv_path = out / "gui_surface_text_catalog.csv"
    validation_path = out / "gui_surface_text_validation.json"
    rows = surface_text_catalog()
    json_path.write_text(json.dumps({"schema": GUI_SURFACE_TEXT_SCHEMA, "texts": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "category", "ja", "en"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in ("key", "category", "ja", "en")})
    validation_path.write_text(json.dumps(validate_surface_text_catalog(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "validation": str(validation_path)}


def _translate_tree_item(item: Any, locale: str) -> int:
    count = 0
    old = str(item.text(0))
    new = translate_surface_text(old, locale=locale, category="tree")
    if new != old:
        item.setText(0, new)
        count += 1
    for index in range(item.childCount()):
        count += _translate_tree_item(item.child(index), locale)
    return count


def _text_for_locale(row: SurfaceText | None, locale: str) -> str:
    if row is None:
        return ""
    if locale == "en":
        return row.en
    return row.ja


__all__ = [
    "GUI_SURFACE_TEXT_SCHEMA",
    "GUI_SURFACE_TEXT_VALIDATION_SCHEMA",
    "apply_gui_surface_texts",
    "gui_surface_message",
    "surface_text_catalog",
    "translate_surface_text",
    "validate_surface_text_catalog",
    "write_surface_text_catalog",
]
