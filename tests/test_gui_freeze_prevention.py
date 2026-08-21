from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from geofem_app.gui.background_tasks import CancellationToken, run_callable_with_token
from geofem_app.gui.accessibility import accessibility_policy_contract
from geofem_app.gui.autosave_io import write_autosave_files
from geofem_app.gui.button_icons import button_icon_catalog, panel_button_icon_catalog
from geofem_app.gui.cad_worker import solve_dimension_constraints_snapshot, split_lines_at_intersections_snapshot
from geofem_app.gui.desktop_layout import resolve_desktop_layout_profile
from geofem_app.gui.design_system import commercial_design_tokens, commercial_gui_stylesheet, resolve_gui_font_pt
from geofem_app.gui.display_quality import resolve_display_quality_policy
from geofem_app.gui.file_preview import preview_text_file, read_json_file_guarded, read_mapping_file_guarded
from geofem_app.gui import font_support
from geofem_app.gui.font_support import apply_preferred_gui_font, preferred_gui_font_inventory
from geofem_app.gui.help_system import documentation_link_catalog, help_catalog, help_policy_contract
from geofem_app.gui.mesh_worker import collect_mesh_quality_violations_snapshot, compare_mesh_quality_improvements_snapshot, generate_auto_geometry_mesh_snapshot
from geofem_app.gui.post_export_worker import (
    audit_post_report_snapshot,
    build_selected_report_snapshot,
    compare_post_image_snapshot,
    copy_report_pdf_snapshot,
    export_scene_pdf_snapshot,
    save_post_image_snapshot,
)
from geofem_app.gui.post_worker import load_post_table_snapshot, load_srm_post_snapshot, materialize_post_component_snapshot
from geofem_app.gui.job_controller import GuiJobController
from geofem_app.gui.recovery_io import compare_recovery_files, read_jsonl_tail, recovery_candidate_infos
from geofem_app.gui.result_paging import read_csv_table_page, result_table_page, summarize_csv_table
from geofem_app.gui.solver_input_policy import apply_gui_solver_output_defaults
from geofem_app.gui.visual_hierarchy import visual_hierarchy_contract
from geofem_app.gui.watchdog import GuiFreezeWatchdogRecord, append_watchdog_record
from geofem_app.samples import plane_strain_quad4_sample


class GuiFreezePreventionTests(unittest.TestCase):
    def test_gui_solver_input_defaults_enable_lazy_reports_without_overriding_explicit_policy(self) -> None:
        data = yaml.safe_load(apply_gui_solver_output_defaults("analysis:\n  type: static_plane_strain\n"))
        self.assertTrue(data["output"]["lazy_reports"])
        self.assertEqual(data["solver"]["execution"]["context"], "gui")
        self.assertEqual(data["solver"]["execution"]["profile"], "interactive")
        self.assertTrue(data["solver"]["srm"]["progress_stdout"])

        explicit_false = yaml.safe_load(apply_gui_solver_output_defaults("output:\n  lazy_reports: false\n"))
        self.assertFalse(explicit_false["output"]["lazy_reports"])

        explicit_defer = yaml.safe_load(apply_gui_solver_output_defaults("output:\n  defer_reports: false\n"))
        self.assertNotIn("lazy_reports", explicit_defer["output"])

        explicit_srm_progress = yaml.safe_load(apply_gui_solver_output_defaults("solver:\n  srm:\n    progress_stdout: false\n"))
        self.assertFalse(explicit_srm_progress["solver"]["srm"]["progress_stdout"])

    def test_main_preview_updates_are_debounced_and_cached_for_large_models(self) -> None:
        main_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        domain_source = Path("geofem_app/gui/domain_panels.py").read_text(encoding="utf-8")
        analysis_source = Path("geofem_app/gui/analysis_panel.py").read_text(encoding="utf-8")

        self.assertIn("self._preview_update_delay_ms = 120", main_source)
        self.assertIn("self.preview_update_timer.stop()", main_source)
        self.assertIn("self.request_preview_update(reset_view=True, reason=message, delay_ms=0)", main_source)
        self.assertIn("def _preview_mesh_from_config", main_source)
        self.assertIn("self._preview_mesh_cache_key", main_source)
        self.assertIn("def _cad_repair_preview_summary", main_source)
        self.assertIn("def _cad_repair_diagnostics_for_geometry", main_source)
        self.assertIn("self._invalidate_cad_repair_preview_cache()", main_source)
        self.assertIn("diagnostics = self._cad_repair_diagnostics_for_geometry()", main_source)
        self.assertIn("def _model_check_cache_signature", main_source)
        self.assertIn("self._model_check_cached_signature", main_source)
        self.assertIn("self._cached_model_check_issues(signature)", main_source)
        self.assertIn("self._finish_solver_preflight_issues(cached, source=\"cache\")", main_source)
        self.assertIn("def _preview_item_selectable", main_source)
        self.assertIn("def _draw_batched_mesh_elements", main_source)
        self.assertIn("def _draw_batched_mesh_nodes", main_source)
        self.assertIn("def _select_nearest_mesh_lod_item", main_source)
        self.assertIn("def _start_preview_mesh_worker_if_needed", main_source)
        self.assertIn("def _cfg_prefers_deferred_load_refresh", main_source)
        self.assertIn("def _show_deferred_model_load_notice", main_source)
        self.assertIn("_last_load_deferred_heavy_refresh", main_source)
        self.assertIn("self._load_cfg(data, keep_yaml=True)", main_source)
        self.assertIn("preview_performance_profile", main_source)
        self.assertIn("analysis_output_policy", main_source)
        self.assertIn("resolve_analysis_output_dir", main_source)
        self.assertIn("_prompt_skip_analysis_conditions_after_yaml_load", main_source)
        self.assertIn("読み込んだ解析条件で、解析実行画面へ進みますか？", main_source)
        self.assertNotIn("解析条件の確認を省略しますか？", main_source)
        self.assertIn("_set_solver_navigation_suppressed(True)", main_source)
        self.assertIn("_apply_solver_navigation_state", main_source)
        self.assertIn("QPainterPath", main_source)
        self.assertIn("selection LOD 1/", main_source)
        self.assertIn("request_preview_update(reset_view=False, reason=\"contour levels\")", domain_source)
        self.assertIn("request_preview_update(reset_view=False, reason=\"axisymmetric guide\")", analysis_source)

    def test_button_icon_catalog_covers_primary_gui_actions(self) -> None:
        catalog = button_icon_catalog()
        required_roles = {
            "view.reset",
            "model.check",
            "draw.line",
            "selection.rectangle",
            "selection.filter",
            "analysis.run",
            "analysis.reset_results",
            "analysis.stop",
            "yaml.sync",
            "project.save",
            "project.lock",
            "project.unlock",
            "error.jump",
            "error.fix_all",
            "recovery.open",
            "audit.open",
        }
        self.assertTrue(required_roles.issubset(catalog))
        self.assertTrue(all(catalog[role]["standard_pixmap"].startswith("SP_") for role in required_roles))
        self.assertTrue(all(catalog[role]["tooltip"] for role in required_roles))
        panel_catalog = panel_button_icon_catalog()
        for panel_key in ("analysis", "geometry", "mesh", "materials", "stages", "results", "report"):
            self.assertIn(panel_key, panel_catalog)
            self.assertTrue(panel_catalog[panel_key].startswith("SP_"))
        help_panels = help_catalog()
        for panel_key in ("analysis", "mesh", "materials", "results", "report"):
            self.assertIn(panel_key, help_panels)
            self.assertTrue(help_panels[panel_key]["help_id"].startswith("geofem.help."))

    def test_large_srm_yaml_load_defers_preview_and_model_check(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg.setdefault("analysis", {})["type"] = "srm"
            cfg.setdefault("mesh", {})["nx"] = 40
            cfg.setdefault("mesh", {})["ny"] = 20
            calls: list[dict[str, object]] = []

            def fail_sync_preview(*_args: object, **kwargs: object) -> None:
                calls.append(dict(kwargs))
                self.fail("large SRM load should not run update_preview synchronously")

            window.update_preview = fail_sync_preview
            window._load_cfg(cfg, keep_yaml=False)
            scene_text = "\n".join(getattr(item, "toPlainText", lambda: "")() for item in window.scene.items())
            self.assertTrue(window._last_load_deferred_heavy_refresh)
            self.assertTrue(window._model_preview_dirty)
            self.assertTrue(window._model_check_dirty)
            self.assertEqual(window._deferred_load_mesh_size, (861, 800))
            self.assertIn("Large model loaded", scene_text)
            self.assertEqual(calls, [])
            checks.append({"mode": window.preview_performance_profile.get("mode")})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"mode": "deferred_load"}])

    def test_commercial_gui_design_tokens_cover_common_states_and_widgets(self) -> None:
        tokens = commercial_design_tokens()
        self.assertEqual(tokens["schema"], "geofem.gui.design_tokens.v1")
        self.assertEqual({"error", "warning", "ok", "info"}, set(tokens["states"]))
        self.assertGreaterEqual(tokens["metrics"]["compact_row_height"], 22)
        self.assertIn("Meiryo", tokens["font_family"])
        self.assertEqual(resolve_gui_font_pt(10, {"GEOFEM_GUI_FONT_SCALE": "1.5"}), 15)
        self.assertEqual(resolve_gui_font_pt(10, {"GEOFEM_GUI_FONT_PT": "18"}), 18)
        stylesheet = commercial_gui_stylesheet()
        for selector in ("QGroupBox", "QPushButton", "QTableWidget", "QStatusBar", "QTabWidget"):
            self.assertIn(selector, stylesheet)
        self.assertIn("font-family", stylesheet)
        self.assertIn("border-left", stylesheet)
        self.assertIn("informationRole", stylesheet)
        for color_name in ("error_surface", "warning_surface", "ok_surface", "selection"):
            self.assertIn(tokens["colors"][color_name], stylesheet)
        accessibility = accessibility_policy_contract()
        self.assertEqual(accessibility["schema"], "geofem.gui.accessibility_policy.v1")
        self.assertIn("QPushButton", accessibility["named_widgets"])
        self.assertIn("border", accessibility["severity_channels"])
        help_policy = help_policy_contract()
        self.assertEqual(help_policy["schema"], "geofem.gui.help_policy.v1")
        self.assertIn("analysis", help_policy["panels"])
        self.assertIn("helpId", help_policy["property_names"])
        documentation_links = documentation_link_catalog()
        for key in ("errors.input_cell", "report.item", "report.audit", "settings.input"):
            self.assertIn(key, documentation_links)
            self.assertTrue(documentation_links[key]["help_url"].startswith("docs/user_guide.md#"))
            self.assertTrue(documentation_links[key]["help_id"].startswith("geofem.help."))
        guide = Path("docs/user_guide.md").read_text(encoding="utf-8")
        for anchor in ("## errors-input-cell", "## report-item", "## report-audit", "## settings-input"):
            self.assertIn(anchor, guide)
        hierarchy = visual_hierarchy_contract()
        self.assertEqual(hierarchy["schema"], "geofem.gui.visual_hierarchy.v1")
        self.assertEqual(hierarchy["roles"]["warning"]["priority"], 0)
        self.assertIn("informationRole", hierarchy["property_names"])

    def test_preferred_gui_font_support_registers_known_local_fonts(self) -> None:
        class FakeFontDatabase:
            families_set: set[str] = set()
            registered: list[str] = []

            @classmethod
            def families(cls) -> list[str]:
                return sorted(cls.families_set)

            @classmethod
            def addApplicationFont(cls, path: str) -> int:
                cls.registered.append(path)
                cls.families_set.update({"Yu Gothic UI", "Meiryo"})
                return len(cls.registered) - 1

            @classmethod
            def applicationFontFamilies(cls, font_id: int) -> list[str]:
                return ["Yu Gothic UI", "Meiryo"] if font_id >= 0 else []

        class FakeApp:
            def __init__(self) -> None:
                self.props: dict[str, object] = {}
                self.font_family = ""

            def setFont(self, font: object) -> None:
                self.font_family = str(font)

            def font(self) -> object:
                class CurrentFont:
                    def family(self) -> str:
                        return "Sans Serif"

                return CurrentFont()

            def setProperty(self, name: str, value: object) -> None:
                self.props[name] = value

        original_files = font_support.KNOWN_JAPANESE_FONT_FILES
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                font_path = Path(tmp_dir) / "fake-font.ttc"
                font_path.write_text("font", encoding="utf-8")
                font_support.KNOWN_JAPANESE_FONT_FILES = (font_path,)
                inventory = preferred_gui_font_inventory(FakeFontDatabase)
                self.assertIn("Yu Gothic UI", inventory["available_preferred"])
                app = FakeApp()
                result = apply_preferred_gui_font(app, FakeFontDatabase, lambda family, size: f"{family}:{size}", 11)
                self.assertEqual(result["status"], "available")
                self.assertEqual(app.props["geofemGuiFontFamily"], "Yu Gothic UI")
                self.assertFalse(app.props["geofemGuiFontCandidatesMissing"])
        finally:
            font_support.KNOWN_JAPANESE_FONT_FILES = original_files

    def test_desktop_layout_profile_scales_for_2560_1440(self) -> None:
        profile = resolve_desktop_layout_profile(2560, 1440)
        self.assertTrue(profile.large_desktop)
        self.assertLessEqual(profile.window_width, 2560)
        self.assertLessEqual(profile.window_height, 1440)
        self.assertGreaterEqual(profile.window_width, 2200)
        self.assertGreaterEqual(profile.window_height, 1250)
        self.assertEqual(sum(profile.horizontal_split_sizes), profile.window_width)
        self.assertEqual(sum(profile.vertical_split_sizes), profile.window_height)
        self.assertGreaterEqual(profile.horizontal_split_sizes[1], 1400)
        self.assertGreaterEqual(profile.vertical_split_sizes[0], 1000)
        self.assertGreaterEqual(profile.model_view_min_size, (900, 620))

        compact = resolve_desktop_layout_profile(1366, 768)
        self.assertFalse(compact.large_desktop)
        self.assertLessEqual(compact.window_width, 1366)
        self.assertLessEqual(compact.window_height, 768)
        self.assertEqual(sum(compact.horizontal_split_sizes), compact.window_width)

    def test_main_window_applies_2560_1440_layout_profile(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QAction
            from PySide6.QtWidgets import QApplication, QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit, QScrollArea, QTableWidget, QTabWidget
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        original_resolver = main_window_module.resolve_desktop_layout_profile
        checks: list[tuple[int, int, tuple[int, int, int]]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = max(windows, key=lambda widget: getattr(getattr(widget, "desktop_layout_profile", None), "window_width", 0))
            profile = window.desktop_layout_profile
            self.assertTrue(profile.large_desktop)
            self.assertIn("geofem.gui.design_tokens.v1", app.styleSheet())
            self.assertGreaterEqual(window.view.minimumWidth(), 900)
            self.assertGreaterEqual(window.tabs.minimumWidth(), 900)
            self.assertGreaterEqual(window.panel_stack.minimumWidth(), 380)
            self.assertEqual(len(window.view_toolbar_rows), 3)
            self.assertTrue(all(row.count() <= 17 for row in window.view_toolbar_rows))
            self.assertTrue(hasattr(window, "gui_operation_mode_combo"))
            self.assertEqual(window.gui_operation_mode_combo.currentData(), "standard")
            self.assertTrue(hasattr(window, "command_search"))
            self.assertGreaterEqual(window.command_search.count(), 8)
            menu_titles = [action.text().replace("&", "") for action in window.menuBar().actions()]
            self.assertIn("操作", menu_titles)
            operation_index = menu_titles.index("操作")
            self.assertLess(operation_index + 1, len(menu_titles))
            self.assertEqual(menu_titles[operation_index + 1], "詳細設定")
            self.assertTrue(hasattr(window, "advanced_settings_menu"))
            self.assertTrue(hasattr(window, "language_menu"))
            self.assertIn("en", window.language_actions)
            self.assertIn("ja", window.language_actions)
            window.language_actions["en"].trigger()
            QApplication.processEvents()
            self.assertEqual(window.gui_locale, "en")
            self.assertEqual(window.gui_language.currentData(), "en")
            self.assertTrue(window.language_actions["en"].isChecked())
            self.assertEqual(window.advanced_settings_menu.title(), "Advanced Settings")
            self.assertEqual(window.language_menu.title(), "Language")
            self.assertEqual(window.command_menu_root.title(), "Operations")
            self.assertIn("Analysis", [action.text().replace("&", "") for action in window.menuBar().actions()])
            self.assertEqual(window.language_actions["ja"].text(), "Japanese")
            self.assertEqual(window.language_actions["en"].text(), "English")
            self.assertEqual(window.operation_mode_label.text(), "Operation Mode")
            self.assertEqual(window.command_search_label.text(), "Command Search")
            self.assertEqual(window.command_run_button.text(), "Run Command")
            self.assertIn("Run section", window.command_run_button.toolTip())
            self.assertEqual(window.primary_action_group.title(), "Run")
            self.assertTrue(window.primary_action_group.isHidden())
            self.assertEqual(window.confirm_action_group.title(), "Check")
            self.assertFalse(window.aux_run_button.isVisible())
            self.assertIsNone(window.quick_access_toolbar)
            self.assertIsNone(window.quick_access_toolbar_label)
            self.assertIsNone(window.command_toolbar)
            self.assertIsNone(window.command_toolbar_label)
            self.assertTrue(hasattr(window, "command_palette_action"))
            self.assertEqual(window.command_palette_action.text().replace("&", ""), "Command Palette...")
            self.assertIn("popup", window.command_palette_action.toolTip())
            self.assertEqual([window.file_actions[key].text().replace("&", "") for key in ("new", "open", "save")], ["New 2D Sample", "Open Input", "Save Input"])
            self.assertEqual([window.file_actions[key].shortcut().toString() for key in ("new", "open", "save")], ["Ctrl+N", "Ctrl+O", "Ctrl+S"])
            self.assertEqual([action.text().replace("&", "") for action in window.menu_file.actions()[:3]], ["New 2D Sample", "Open Input", "Save Input"])
            self.assertTrue(window.command_menu_root.actions()[0] is window.command_palette_action)
            self.assertEqual(window.command_menu_root.actions()[1].isSeparator(), True)
            self.assertEqual(window.menu_help.title(), "Help")
            version_action = next(action for action in window.menu_help.actions() if action.property("commandId") == "help.version")
            self.assertEqual(version_action.text().replace("&", ""), "Version Info")

            captured_version_dialogs: list[tuple[str, str]] = []

            class FakeMessageBox:
                @staticmethod
                def information(_parent: object, title: str, text: str, *_args: object, **_kwargs: object) -> int:
                    captured_version_dialogs.append((title, text))
                    return 0

            original_post_qt = window._post_report_controller_qt
            try:
                window._select_tree_panel("loads")
                QApplication.processEvents()
                before_tabs_widget = window.tabs.currentWidget()
                before_panel_key = window.current_panel_key
                window._post_report_controller_qt = lambda: {**original_post_qt(), "QMessageBox": FakeMessageBox}
                version_action.trigger()
                QApplication.processEvents()
                self.assertEqual(before_panel_key, "loads")
                self.assertEqual(window.current_panel_key, "loads")
                self.assertIs(window.tabs.currentWidget(), before_tabs_widget)
                self.assertEqual(captured_version_dialogs[0][0], "Version Information")
                self.assertIn("Version:", captured_version_dialogs[0][1])
                self.assertIn("Dependencies:", captured_version_dialogs[0][1])
            finally:
                window._post_report_controller_qt = original_post_qt

            command_palette_calls: list[tuple[str, str, tuple[str, ...]]] = []

            class FakeCommandDialog:
                @staticmethod
                def getItem(_parent: object, title: str, label: str, items: list[str], *_args: object, **_kwargs: object) -> tuple[str, bool]:
                    command_palette_calls.append((title, label, tuple(items)))
                    refresh_label = next(item for item in items if item.startswith("Refresh Workflow Check"))
                    return refresh_label, True

            before_panel_key = window.current_panel_key
            before_tabs_widget = window.tabs.currentWidget()
            window.open_command_palette_dialog(dialog_cls=FakeCommandDialog)
            QApplication.processEvents()
            self.assertEqual(command_palette_calls[0][0], "Command Palette")
            self.assertEqual(command_palette_calls[0][1], "Command")
            self.assertEqual(window.current_panel_key, before_panel_key)
            self.assertIs(window.tabs.currentWidget(), before_tabs_widget)
            self.assertIn("workflow.refresh", window.cfg["gui"]["command_palette"]["recent"])

            cjk = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")

            def visible_japanese_texts() -> list[str]:
                leftovers: list[str] = []
                for action in window.menuBar().actions():
                    text = action.text().replace("&", "")
                    if cjk.search(text):
                        leftovers.append(f"menu:{text}")
                    menu = action.menu()
                    if menu is None:
                        continue
                    for child_action in menu.actions():
                        child_text = child_action.text().replace("&", "")
                        if child_text and cjk.search(child_text):
                            leftovers.append(f"{menu.title()}:{child_text}")
                        child_menu = child_action.menu()
                        if child_menu is None:
                            continue
                        for grandchild_action in child_menu.actions():
                            grandchild_text = grandchild_action.text().replace("&", "")
                            if grandchild_text and cjk.search(grandchild_text):
                                leftovers.append(f"{child_menu.title()}:{grandchild_text}")
                for cls in (QAbstractButton, QLabel):
                    for widget in window.findChildren(cls):
                        if not widget.isVisible():
                            continue
                        text = widget.text().replace("&", "")
                        if text and cjk.search(text):
                            leftovers.append(f"{widget.__class__.__name__}:{text}")
                        tooltip = widget.toolTip().replace("&", "")
                        if tooltip and cjk.search(tooltip):
                            leftovers.append(f"{widget.__class__.__name__}.tooltip:{tooltip}")
                for group in window.findChildren(QGroupBox):
                    if not group.isVisible():
                        continue
                    title = group.title().replace("&", "")
                    if title and cjk.search(title):
                        leftovers.append(f"QGroupBox:{title}")
                    tooltip = group.toolTip().replace("&", "")
                    if tooltip and cjk.search(tooltip):
                        leftovers.append(f"QGroupBox.tooltip:{tooltip}")
                for combo in window.findChildren(QComboBox):
                    if not combo.isVisible():
                        continue
                    tooltip = combo.toolTip().replace("&", "")
                    if tooltip and cjk.search(tooltip):
                        leftovers.append(f"QComboBox.tooltip:{tooltip}")
                    if combo.lineEdit() is not None:
                        placeholder = combo.lineEdit().placeholderText().replace("&", "")
                        if placeholder and cjk.search(placeholder):
                            leftovers.append(f"QComboBox.placeholder:{placeholder}")
                    for index in range(combo.count()):
                        text = combo.itemText(index)
                        if text and cjk.search(text):
                            leftovers.append(f"QComboBox:{text}")
                for line_edit in window.findChildren(QLineEdit):
                    if not line_edit.isVisible():
                        continue
                    placeholder = line_edit.placeholderText().replace("&", "")
                    if placeholder and cjk.search(placeholder):
                        leftovers.append(f"QLineEdit.placeholder:{placeholder}")
                    tooltip = line_edit.toolTip().replace("&", "")
                    if tooltip and cjk.search(tooltip):
                        leftovers.append(f"QLineEdit.tooltip:{tooltip}")
                for tabs in window.findChildren(QTabWidget):
                    if not tabs.isVisible():
                        continue
                    for index in range(tabs.count()):
                        text = tabs.tabText(index)
                        if text and cjk.search(text):
                            leftovers.append(f"QTabWidget:{text}")
                        tooltip = tabs.tabToolTip(index).replace("&", "")
                        if tooltip and cjk.search(tooltip):
                            leftovers.append(f"QTabWidget.tooltip:{tooltip}")
                for table in window.findChildren(QTableWidget):
                    if not table.isVisible():
                        continue
                    tooltip = table.toolTip().replace("&", "")
                    if tooltip and cjk.search(tooltip):
                        leftovers.append(f"QTableWidget.tooltip:{tooltip}")
                    for column in range(table.columnCount()):
                        header = table.horizontalHeaderItem(column)
                        text = header.text().replace("&", "") if header is not None else ""
                        if text and cjk.search(text):
                            leftovers.append(f"QTableWidget.header:{text}")
                    for row in range(table.rowCount()):
                        header = table.verticalHeaderItem(row)
                        text = header.text().replace("&", "") if header is not None else ""
                        if text and cjk.search(text):
                            leftovers.append(f"QTableWidget.vheader:{text}")
                    for row in range(min(table.rowCount(), 8)):
                        for column in range(min(table.columnCount(), 12)):
                            item = table.item(row, column)
                            if item is None:
                                continue
                            text = item.text().replace("&", "")
                            if text and cjk.search(text):
                                leftovers.append(f"QTableWidget.cell:{text}")
                            tooltip = item.toolTip().replace("&", "")
                            if tooltip and cjk.search(tooltip):
                                leftovers.append(f"QTableWidget.cell.tooltip:{tooltip}")
                for action in window.findChildren(QAction):
                    for kind, value in (
                        ("QAction", action.text()),
                        ("QAction.tooltip", action.toolTip()),
                        ("QAction.status", action.statusTip()),
                    ):
                        text = value.replace("&", "")
                        if text and cjk.search(text):
                            leftovers.append(f"{kind}:{text}")
                return sorted(set(leftovers))

            for panel in ("analysis", "geometry", "mesh", "materials", "boundary_conditions", "loads", "stages", "model_check", "solver", "results", "report"):
                window._select_tree_panel(panel)
                QApplication.processEvents()
                self.assertEqual(visible_japanese_texts(), [], panel)
            window._select_tree_panel("analysis")
            QApplication.processEvents()
            self.assertTrue(window.workspace_tabs_caption.isHidden())
            self.assertEqual(window.workspace_tabs_caption.maximumHeight(), 0)
            self.assertTrue(window.tabs.tabBar().isHidden())
            self.assertTrue(window.tabs.property("workspaceViewTabsDeprecated"))
            self.assertFalse(hasattr(window, "aux_open_detail_button"))
            self.assertFalse(hasattr(window, "aux_model_check_button"))
            self.assertIn("Advanced", window.tabs.tabToolTip(window.tabs.indexOf(window.workspace_tab_widgets["detail"])))
            self.assertIn("Standard", [window.gui_operation_mode_combo.itemText(index) for index in range(window.gui_operation_mode_combo.count())])
            self.assertEqual([window.display_quality_mode.itemText(index) for index in range(window.display_quality_mode.count())], ["Auto", "Full", "Fast"])
            self.assertEqual(window.aux_panel_title.text(), "Analysis")
            self.assertIn("Optional setup", window.aux_panel_stats.text())
            self.assertEqual(window.aux_selection_context_group.title(), "Current")
            self.assertEqual(window.aux_current_task_label.text(), "Current Task: Analysis")
            self.assertIn("Selected: Nodes 0", window.aux_current_selection_label.text())
            self.assertIn("Right click: disabled", window.aux_context_menu_status_label.text())
            self.assertEqual(window.aux_summary_group.title(), "Status")
            self.assertTrue(window.aux_action_group.isHidden())
            self.assertEqual(window.aux_action_group.maximumHeight(), 0)
            self.assertEqual(window.auxiliary_panel_content.layout().indexOf(window.aux_action_group), -1)
            self.assertEqual(window.aux_result_control_group.title(), "Result Display")
            self.assertIn("ERROR", window.statusBar().currentMessage())
            window.language_actions["ja"].trigger()
            QApplication.processEvents()
            self.assertEqual(window.gui_locale, "ja")
            self.assertEqual(window.gui_language.currentData(), "ja")
            self.assertTrue(window.language_actions["ja"].isChecked())
            self.assertEqual(window.advanced_settings_menu.title(), "詳細設定")
            self.assertEqual(window.language_menu.title(), "言語")
            self.assertEqual(window.command_menu_root.title(), "操作")
            restored_menu_titles = [action.text().replace("&", "") for action in window.menuBar().actions()]
            self.assertIn("解析", restored_menu_titles)
            self.assertNotIn("解析条件", restored_menu_titles)
            self.assertEqual(window.language_actions["ja"].text(), "日本語")
            self.assertEqual(window.language_actions["en"].text(), "英語")
            self.assertEqual(window.operation_mode_label.text(), "操作モード")
            self.assertEqual(window.command_search_label.text(), "コマンド検索")
            self.assertEqual(window.command_run_button.text(), "コマンド実行")
            self.assertIn("下部の実行", window.command_run_button.toolTip())
            self.assertEqual(window.primary_action_group.title(), "実行")
            self.assertTrue(window.primary_action_group.isHidden())
            self.assertEqual(window.confirm_action_group.title(), "確認")
            self.assertFalse(window.aux_run_button.isVisible())
            self.assertIsNone(window.quick_access_toolbar)
            self.assertIsNone(window.quick_access_toolbar_label)
            self.assertIsNone(window.command_toolbar)
            self.assertIsNone(window.command_toolbar_label)
            self.assertEqual(window.command_palette_action.text().replace("&", ""), "コマンドパレット...")
            self.assertIn("ポップアップ", window.command_palette_action.toolTip())
            self.assertEqual([window.file_actions[key].text().replace("&", "") for key in ("new", "open", "save")], ["新規2Dサンプル", "入力を開く", "入力を保存"])
            self.assertEqual([action.text().replace("&", "") for action in window.menu_file.actions()[:3]], ["新規2Dサンプル", "入力を開く", "入力を保存"])
            self.assertTrue(window.command_menu_root.actions()[0] is window.command_palette_action)
            self.assertEqual(window.command_menu_root.actions()[1].isSeparator(), True)
            self.assertTrue(window.workspace_tabs_caption.isHidden())
            self.assertEqual(window.workspace_tabs_caption.maximumHeight(), 0)
            self.assertTrue(window.tabs.tabBar().isHidden())
            self.assertEqual(window.tabs.tabText(window.tabs.indexOf(window.workspace_tab_widgets["model"])), "モデル表示")
            self.assertFalse(window.tabs.isTabVisible(window.tabs.indexOf(window.workspace_tab_widgets["audit"])))
            detail_index = window.gui_operation_mode_combo.findData("detail")
            window.gui_operation_mode_combo.setCurrentIndex(detail_index)
            QApplication.processEvents()
            self.assertTrue(window.tabs.isTabVisible(window.tabs.indexOf(window.workspace_tab_widgets["detail"])))
            standard_index = window.gui_operation_mode_combo.findData("standard")
            window.gui_operation_mode_combo.setCurrentIndex(standard_index)
            QApplication.processEvents()
            self.assertIn("標準", [window.gui_operation_mode_combo.itemText(index) for index in range(window.gui_operation_mode_combo.count())])
            self.assertEqual([window.display_quality_mode.itemText(index) for index in range(window.display_quality_mode.count())], ["自動", "高品質", "高速"])
            self.assertEqual(window.aux_panel_title.text(), "解析条件")
            self.assertIn("任意設定", window.aux_panel_stats.text())
            self.assertEqual(window.aux_summary_group.title(), "状況")
            self.assertTrue(window.aux_action_group.isHidden())
            self.assertEqual(window.aux_action_group.maximumHeight(), 0)
            self.assertEqual(window.auxiliary_panel_content.layout().indexOf(window.aux_action_group), -1)
            self.assertEqual(window.aux_result_control_group.title(), "結果表示調整")
            self.assertIn("エラー", window.statusBar().currentMessage())
            yaml_command_index = window.command_search.findData("yaml.open")
            self.assertGreaterEqual(yaml_command_index, 0)
            window.command_search.setCurrentIndex(yaml_command_index)
            window.pin_selected_command()
            self.assertIn("yaml.open", window.cfg["gui"]["command_palette"]["pinned"])
            self.assertEqual(window.tree.topLevelItem(0).childCount(), 4)
            self.assertTrue(hasattr(window, "right_panel_stack"))
            self.assertEqual(window.aux_selection_context_group.title(), "現在")
            self.assertIn("現在の作業:", window.aux_current_task_label.text())
            self.assertIn("選択中:", window.aux_current_selection_label.text())
            self.assertIn("右クリック:", window.aux_context_menu_status_label.text())
            self.assertIs(window.right_panel_stack.currentWidget(), window.auxiliary_panel)
            self.assertTrue(window.selection_builder_toolbar_widget.isHidden())
            window.execute_selected_command()
            QApplication.processEvents()
            self.assertEqual(window.gui_operation_mode_combo.currentData(), "detail")
            self.assertIs(window.right_panel_stack.currentWidget(), window.auxiliary_panel)
            self.assertIs(window.tabs.currentWidget(), window.yaml_editor)
            detail_index = window.gui_operation_mode_combo.findData("detail")
            self.assertGreaterEqual(detail_index, 0)
            window.gui_operation_mode_combo.setCurrentIndex(detail_index)
            QApplication.processEvents()
            self.assertIs(window.right_panel_stack.currentWidget(), window.auxiliary_panel)
            self.assertGreaterEqual(window.tree.topLevelItem(0).childCount(), 4)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertFalse(window.selection_builder_toolbar_widget.isHidden())
            standard_index = window.gui_operation_mode_combo.findData("standard")
            window.gui_operation_mode_combo.setCurrentIndex(standard_index)
            QApplication.processEvents()
            self.assertIs(window.right_panel_stack.currentWidget(), window.auxiliary_panel)
            self.assertTrue(window.selection_builder_toolbar_widget.isHidden())
            standard_geometry_sizes = window.workspace_splitter.sizes()
            self.assertLessEqual(standard_geometry_sizes[0], profile.horizontal_split_sizes[0])
            self.assertLessEqual(standard_geometry_sizes[2], profile.horizontal_split_sizes[2])
            self.assertGreater(standard_geometry_sizes[1], profile.horizontal_split_sizes[1])
            self.assertTrue(window.view_toolbar_selection_widget.isHidden())
            self.assertTrue(hasattr(window, "workflow_ribbon"))
            self.assertEqual(window.workflow_ribbon.property("placement"), "model_top_left")
            self.assertLessEqual(window.workflow_ribbon.maximumHeight(), 116)
            self.assertFalse(window.workflow_ribbon_progress_label.isVisible())
            self.assertTrue(hasattr(window, "workflow_ribbon_back_button"))
            self.assertTrue(window.workflow_ribbon_back_button.text())
            self.assertTrue(hasattr(window, "workflow_ribbon_step_buttons"))
            self.assertTrue(window.workflow_ribbon_step_buttons)
            self.assertTrue(all(button.minimumHeight() >= 24 for button in window.workflow_ribbon_step_buttons))
            self.assertTrue(all(button.maximumHeight() <= 44 for button in window.workflow_ribbon_step_buttons))
            self.assertTrue(hasattr(window, "workflow_ribbon_next_label"))
            self.assertTrue(window.workflow_ribbon_next_label.text())
            self.assertIsNone(window.workflow_ribbon_step_scroll)
            self.assertTrue(hasattr(window, "workflow_scroll_area"))
            self.assertTrue(hasattr(window, "workflow_next_step_label"))
            self.assertTrue(hasattr(window, "workflow_selected_detail_label"))
            self.assertTrue(all(isinstance(page, QScrollArea) for page in window.panel_pages.values()))
            self.assertTrue(hasattr(window, "stage_workspace_tabs"))
            self.assertEqual(
                [window.stage_workspace_tabs.tabText(index) for index in range(window.stage_workspace_tabs.count())],
                ["詳細フォーム", "差分/承認", "内部データ"],
            )
            self.assertLessEqual(window.workflow_guidance_table.columnCount(), 4)
            self.assertLessEqual(window.workflow_guidance_table.maximumHeight(), 240)
            self.assertGreaterEqual(window.workflow_guidance_table.currentRow(), 0)
            self.assertTrue(window.workflow_selected_detail_label.text())
            workflow_buttons = window.workflow_scroll_area.findChildren(type(window.workflow_refresh_button))
            self.assertTrue(workflow_buttons)
            self.assertTrue(all(button.minimumHeight() >= 36 for button in workflow_buttons))
            icon_buttons = {
                button.property("iconRole"): button
                for button in window.findChildren(type(window.workflow_refresh_button))
                if button.property("iconRole")
            }
            for role in ("view.reset", "model.check", "analysis.run", "analysis.stop", "project.save", "error.jump"):
                self.assertIn(role, icon_buttons)
                self.assertFalse(icon_buttons[role].icon().isNull())
                self.assertTrue(icon_buttons[role].accessibleName())
                self.assertTrue(icon_buttons[role].accessibleDescription())
                self.assertEqual(icon_buttons[role].focusPolicy(), Qt.FocusPolicy.StrongFocus)
                self.assertTrue(str(icon_buttons[role].property("helpId")).startswith("geofem.help."))
                self.assertIn("Help:", icon_buttons[role].toolTip())
            for panel_key in ("analysis", "geometry", "mesh", "materials", "results", "report"):
                page = window.panel_pages[panel_key]
                panel_buttons = page.findChildren(type(window.workflow_refresh_button))
                self.assertTrue(panel_buttons, panel_key)
                self.assertTrue(all(button.minimumHeight() >= 32 for button in panel_buttons), panel_key)
                self.assertTrue(all(button.property("iconRole") for button in panel_buttons), panel_key)
                self.assertTrue(all(not button.icon().isNull() for button in panel_buttons), panel_key)
                self.assertTrue(all(button.accessibleName() for button in panel_buttons), panel_key)
                self.assertTrue(all(button.focusPolicy() == Qt.FocusPolicy.StrongFocus for button in panel_buttons), panel_key)
                self.assertTrue(all(str(button.property("helpId")).startswith("geofem.help.") for button in panel_buttons), panel_key)
                self.assertTrue(all(str(button.property("helpUrl")).startswith("docs/user_guide.md#") for button in panel_buttons), panel_key)
                self.assertIn(panel_key, window.help_policy_results)
                self.assertGreater(window.help_policy_results[panel_key].linked_count, 0)
            window._activate_panel("stages")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertTrue(window.inline_detail_sheet_group.isVisible())
            self.assertTrue(window.aux_stage_action_group.isHidden())
            self.assertTrue(window.stage_standard_action_bar.isVisible())
            self.assertTrue(window.stage_change_actions_widget.isHidden())
            stage_buttons = [
                button
                for button in window.panel_pages["stages"].findChildren(type(window.workflow_refresh_button))
                if button.property("stageSheetButton")
            ]
            self.assertTrue(stage_buttons)
            stage_list_buttons = [button for button in stage_buttons if button.property("stageSheetButtonGroup") == "stage_list"]
            stage_change_buttons = [button for button in stage_buttons if button.property("stageSheetButtonGroup") == "stage_change"]
            self.assertFalse(stage_list_buttons)
            self.assertTrue(stage_change_buttons)
            self.assertTrue(all(button.property("uniformPanelButtonColumns") == 4 for button in stage_change_buttons))
            right_stage_list_buttons = [
                button
                for button in window.aux_stage_action_group.findChildren(type(window.workflow_refresh_button))
                if button.property("stageSheetButtonGroup") == "stage_list"
            ]
            self.assertTrue(right_stage_list_buttons)
            self.assertTrue(all(button.property("uniformPanelButtonColumns") == 2 for button in right_stage_list_buttons))
            window._activate_panel("model_check")
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["model_check"])
            self.assertGreaterEqual(window.model_check_workflow_table.rowCount(), 7)
            self.assertTrue(window.aux_confirm_action_group.isVisible())
            self.assertTrue(window.confirm_action_group.isHidden())
            confirm_buttons = window.aux_confirm_action_group.findChildren(type(window.workflow_refresh_button))
            self.assertTrue(confirm_buttons)
            self.assertTrue(all(button.property("uniformPanelButtonColumns") == 2 for button in confirm_buttons))
            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
            window._activate_panel("solver")
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["solver"])
            self.assertTrue(window.aux_solver_action_group.isVisible())
            self.assertTrue(window.primary_action_group.isHidden())
            solver_buttons = window.aux_solver_action_group.findChildren(type(window.workflow_refresh_button))
            self.assertTrue(solver_buttons)
            self.assertTrue(all(button.property("uniformPanelButtonColumns") == 2 for button in solver_buttons))
            self.assertGreaterEqual(window.solver_final_summary_table.rowCount(), 12)
            solver_summary = {
                window.solver_final_summary_table.item(row, 0).text(): (
                    window.solver_final_summary_table.item(row, 1).text(),
                    window.solver_final_summary_table.item(row, 2).text(),
                )
                for row in range(window.solver_final_summary_table.rowCount())
            }
            self.assertIn("1件", solver_summary["材料数"][0])
            self.assertEqual(solver_summary["未割当"][0], "0件")
            self.assertIn("2件", solver_summary["境界条件数"][0])
            self.assertIn("1件", solver_summary["荷重条件数"][0])
            self.assertIn("単一静的解析", solver_summary["ステージ数"][0])
            self.assertIn("要素", solver_summary["メッシュ品質"][0])
            self.assertIn("small_deformation", solver_summary["変形/高速化"][0])
            self.assertIn("フォールバック", solver_summary)
            self.assertIn("後処理方針", solver_summary)
            setting_widgets = (
                window.panel_pages["analysis"].findChildren(type(window.selection_expr_value))
                + window.panel_pages["analysis"].findChildren(type(window.display_quality_mode))
                + window.panel_pages["results"].findChildren(type(window.result_view))
                + window.panel_pages["report"].findChildren(type(window.result_view))
            )
            self.assertTrue(setting_widgets)
            self.assertTrue(all(str(widget.property("helpUrl")).startswith("docs/user_guide.md#") for widget in setting_widgets))
            self.assertTrue(all("Help:" in widget.toolTip() for widget in setting_widgets))
            window.result_table.setRowCount(1)
            window.result_table.setColumnCount(1)
            window.mark_table_cell_error(window.result_table, 0, 0, "value", "sample error")
            error_item = window.cell_error_table.item(0, 0)
            self.assertIsNotNone(error_item)
            error_payload = error_item.data(Qt.ItemDataRole.UserRole)
            self.assertTrue(str(error_payload["help_url"]).startswith("docs/user_guide.md#errors-input-cell"))
            self.assertIn("Help: geofem.help.errors.input_cell", error_item.toolTip())
            window.add_report_text_block("Help Link", "report item")
            report_item = window.report_page_table.item(0, 0)
            self.assertIsNotNone(report_item)
            report_payload = report_item.data(Qt.ItemDataRole.UserRole)
            self.assertTrue(str(report_payload["help_url"]).startswith("docs/user_guide.md#report-item"))
            self.assertEqual(str(window.report_page_table.property("helpUrl")), "docs/user_guide.md#report-item")
            self.assertEqual(window.view.property("informationRole"), "primary")
            self.assertEqual(window.result_view.property("informationRole"), "primary")
            self.assertEqual(window.result_table.property("informationRole"), "primary")
            self.assertEqual(window.results_summary.property("informationRole"), "primary")
            self.assertEqual(window.report_summary.property("informationRole"), "primary")
            self.assertEqual(window.check_table.property("informationRole"), "warning")
            self.assertEqual(window.cell_error_table.property("informationRole"), "warning")
            self.assertEqual(window.check_summary.property("informationRole"), "warning")
            self.assertEqual(window.log.property("informationRole"), "detail")
            self.assertEqual(window.audit_log_table.property("informationRole"), "detail")
            self.assertIn("results", window.visual_hierarchy_results)
            self.assertGreater(window.visual_hierarchy_results["results"].layered_count, 0)
            self.assertIn("Information layer:", window.result_view.accessibleDescription())
            self.assertIn("Information layer:", window.log.accessibleDescription())
            self.assertGreaterEqual(window.accessibility_policy_result.focusable_count, 20)
            self.assertGreaterEqual(window.accessibility_policy_result.named_count, 20)
            checks.append((profile.window_width, profile.window_height, profile.horizontal_split_sizes))
            for candidate in windows:
                candidate.close()
            return 0

        QApplication.exec = fake_exec
        main_window_module.resolve_desktop_layout_profile = lambda _width, _height: original_resolver(2560, 1440)
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
            main_window_module.resolve_desktop_layout_profile = original_resolver

        self.assertEqual(len(checks), 1)
        width, height, split = checks[0]
        self.assertGreaterEqual(width, 2200)
        self.assertGreaterEqual(height, 1250)
        self.assertGreaterEqual(split[1], 1400)

    def test_main_window_compact_profile_hides_low_priority_controls(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        original_resolver = main_window_module.resolve_desktop_layout_profile
        checks: list[tuple[int, int, list[int]]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            profile = window.desktop_layout_profile
            self.assertFalse(profile.large_desktop)
            self.assertGreaterEqual(profile.tree_min_width, 248)
            self.assertGreaterEqual(profile.panel_min_width, 330)
            detail_index = window.gui_operation_mode_combo.findData("detail")
            self.assertGreaterEqual(detail_index, 0)
            window.gui_operation_mode_combo.setCurrentIndex(detail_index)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertEqual(window.workflow_ribbon.property("placement"), "model_top_left")
            self.assertFalse(window.workflow_ribbon_progress_label.isVisible())
            self.assertTrue(window.workflow_ribbon_back_button.isVisible())
            self.assertTrue(window.workflow_ribbon_jump_button.isVisible())
            window._set_solver_navigation_suppressed(True)
            QApplication.processEvents()
            self.assertFalse(window.workflow_ribbon_back_button.isVisible())
            self.assertFalse(window.workflow_ribbon_jump_button.isVisible())
            self.assertFalse(window.workflow_back_button.isVisible())
            self.assertFalse(window.workflow_jump_button.isVisible())
            self.assertTrue(all(not button.isEnabled() for button in window.workflow_ribbon_step_buttons))
            window._set_solver_navigation_suppressed(False)
            window.refresh_workflow_guidance()
            QApplication.processEvents()
            self.assertTrue(window.workflow_ribbon_back_button.isVisible())
            self.assertTrue(window.workflow_ribbon_jump_button.isVisible())
            back_geometry = window.workflow_ribbon_back_button.geometry().getRect()
            jump_geometry = window.workflow_ribbon_jump_button.geometry().getRect()
            window.workflow_ribbon_jump_button.click()
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "mesh")
            self.assertEqual(window.workflow_ribbon_back_button.geometry().getRect(), back_geometry)
            self.assertEqual(window.workflow_ribbon_jump_button.geometry().getRect(), jump_geometry)
            window.workflow_ribbon_back_button.click()
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertEqual(window.workflow_ribbon_back_button.geometry().getRect(), back_geometry)
            self.assertEqual(window.workflow_ribbon_jump_button.geometry().getRect(), jump_geometry)
            self.assertTrue(window.detail_action_group.isHidden())
            self.assertTrue(window.project_action_group.isHidden())
            self.assertTrue(window.confirm_action_group.isHidden())
            self.assertEqual(window.primary_action_group.title(), "実行")
            self.assertTrue(window.primary_action_group.isHidden())
            self.assertTrue(window.view_toolbar_display_widget.isHidden())
            self.assertTrue(window.selection_builder_toolbar_widget.isHidden())
            self.assertNotIn("[", window.command_search.currentText())
            self.assertNotIn("Ctrl", window.command_search.currentText())
            primary_texts = {button.text() for button in window.primary_action_group.findChildren(QPushButton)}
            self.assertEqual({"解析実行", "結果リセット", "停止", "保存"}, primary_texts)
            self.assertIs(window.right_panel_stack.currentWidget(), window.auxiliary_panel)
            self.assertIn("詳細:", window.aux_panel_summary.text())
            self.assertIn("選択:", window.aux_panel_stats.text())
            checks.append((profile.window_width, profile.window_height, window.workspace_splitter.sizes()))
            window.close()
            return 0

        QApplication.exec = fake_exec
        main_window_module.resolve_desktop_layout_profile = lambda _width, _height: original_resolver(1366, 768)
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
            main_window_module.resolve_desktop_layout_profile = original_resolver

        self.assertEqual(len(checks), 1)
        width, height, split = checks[0]
        self.assertLessEqual(width, 1366)
        self.assertLessEqual(height, 768)
        self.assertGreaterEqual(split[0], 260)
        self.assertGreaterEqual(split[2], 330)

    def test_solver_finish_message_reports_elapsed_time(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "run_20260528_010203"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "name": "srm",
                                "solver": {
                                    "srm": {
                                        "factor_of_safety": 1.25,
                                        "stable_factor": 1.25,
                                        "failed_factor": 1.5,
                                        "search_mode": "linear",
                                        "trials": [
                                            {"factor": 1.25, "ok": True, "converged": True, "plastic_ratio": 0.0},
                                            {"factor": 1.5, "ok": False, "converged": True, "plastic_ratio": 1.0, "failure_reason": "plastic_divergence"},
                                        ],
                                    }
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_exec(app: QApplication) -> int:
                windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
                self.assertTrue(windows)
                window = windows[-1]
                window.project_root = root
                self.assertEqual(window._format_solver_elapsed(125.4), "2分05秒")
                message = window._solver_completion_message(0, 125.4, run_dir)
                self.assertIn("解析が終了しました", message)
                self.assertIn("経過時間: 2分05秒", message)
                self.assertIn(str(results_dir), message)
                self.assertIn("SRM FOS: srm: FOS=1.25", window._solver_completion_srm_text(results_dir / "summary.json"))
                window.gui_locale = "en"
                english_message = window._solver_completion_message(0, 125.4, run_dir)
                self.assertIn("SRM FOS: srm: FOS=1.25", english_message)
                window.gui_locale = "ja"
                captured: list[tuple[int, float, Path]] = []
                window._show_solver_completion_dialog = lambda code, elapsed, path: captured.append((code, elapsed, path))
                window._solver_started_perf = time.perf_counter() - 65.0
                window._finished(0, run_dir)
                self.assertEqual(captured[0][0], 0)
                self.assertGreaterEqual(captured[0][1], 60.0)
                self.assertEqual(captured[0][2], run_dir)
                self.assertIsNotNone(window.last_solver_elapsed_seconds)
                self.assertIn("最新結果", window.results_summary.text())
                self.assertEqual(window.result_judgment_model["kind"], "srm")
                self.assertIn("FOS=1.25", window.result_judgment_headline.text())
                judgment_metrics = {
                    caption.text(): value.text()
                    for caption, value in zip(window.result_judgment_metric_captions, window.result_judgment_metric_values)
                }
                self.assertEqual(judgment_metrics["判定区間"], "1.25 - 1.5")
                self.assertEqual(judgment_metrics["試行数"], "2")
                log_text = window.log.toPlainText()
                self.assertIn("[GUI][SRM] srm: FOS=1.25", log_text)
                self.assertIn("[GUI][SRM trial] srm #2: factor=1.5", log_text)
                checks.append({"captured": bool(captured)})
                window.close()
                return 0

            QApplication.exec = fake_exec
            try:
                self.assertEqual(main_window_module.run_gui(), 0)
            finally:
                QApplication.exec = original_exec

        self.assertEqual(checks, [{"captured": True}])

    def test_startup_right_pane_tracks_analysis_conditions(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]

            def tree_labels(item: object) -> list[str]:
                labels = [item.text(0)]
                for index in range(item.childCount()):
                    labels.extend(tree_labels(item.child(index)))
                return labels

            self.assertEqual(window.current_panel_key, "analysis")
            self.assertEqual(window.tree.currentItem().text(0), "解析条件")
            self.assertNotIn("作業ガイド", tree_labels(window.tree.topLevelItem(0)))
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["analysis"])
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertTrue(window.model_cad_palette_widget.isHidden())
            self.assertTrue(window.model_mesh_palette_widget.isHidden())
            self.assertEqual(window.primary_action_group.title(), "実行")
            self.assertTrue(window.primary_action_group.isHidden())
            self.assertEqual(window.confirm_action_group.title(), "確認")
            self.assertTrue(window.confirm_action_group.isHidden())
            self.assertFalse(hasattr(window, "aux_open_detail_button"))
            self.assertFalse(hasattr(window, "aux_model_check_button"))
            self.assertTrue(window.workspace_tabs_caption.isHidden())
            self.assertTrue(window.tabs.tabBar().isHidden())
            profile = window.desktop_layout_profile
            expected_analysis_width = max(profile.center_min_width, profile.model_view_min_size[0])
            analysis_page = window.panel_pages["analysis"]
            self.assertTrue(analysis_page.property("alignedWithModelWorkspace"))
            self.assertGreaterEqual(analysis_page.minimumWidth(), expected_analysis_width)
            self.assertGreaterEqual(analysis_page.minimumHeight(), profile.model_view_min_size[1])
            window.open_current_detail_panel()
            QApplication.processEvents()
            self.assertEqual(window._current_gui_operation_mode(), "standard")
            self.assertEqual(window.current_panel_key, "analysis")
            self.assertEqual(window._detail_return_panel, "analysis")
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["analysis"])
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertTrue(window.model_cad_palette_widget.isHidden())
            self.assertTrue(window.model_mesh_palette_widget.isHidden())
            self.assertTrue(window.view_toolbar_model_widget.isHidden())
            self.assertTrue(window.view_toolbar_display_widget.isHidden())
            self.assertTrue(window.detail_action_group.isHidden())
            window._select_tree_panel("analysis")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "analysis")
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["analysis"])
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertTrue(window.model_cad_palette_widget.isHidden())
            self.assertTrue(window.model_mesh_palette_widget.isHidden())
            analysis_split_sizes = list(window.workspace_splitter.sizes())
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertIsNot(window.tabs.currentWidget(), window.panel_stack)
            self.assertTrue(window.model_cad_palette_widget.isVisible())
            geometry_split_sizes = list(window.workspace_splitter.sizes())
            self.assertEqual(analysis_split_sizes[1], geometry_split_sizes[1])
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_analysis_panel_loads_input_templates_without_changing_project_root(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]

            def scene_kinds() -> set[str]:
                return {
                    str(data.get("kind"))
                    for data in (item.data(0) for item in window.scene.items())
                    if isinstance(data, dict) and data.get("kind")
                }

            window._select_tree_panel("analysis")
            QApplication.processEvents()

            self.assertTrue(hasattr(window, "input_template_combo"))
            combo_entries = [str(window.input_template_combo.itemText(index)) for index in range(window.input_template_combo.count())]
            combo_ids = [str(window.input_template_combo.itemData(index)) for index in range(window.input_template_combo.count())]
            self.assertTrue(any("標準2Dサンプル" in label for label in combo_entries))
            self.assertTrue(any("plane_strain_quad4_bbar" in value for value in combo_ids))
            srm_variant_ids = [value for value in combo_ids if "slope_srm_drucker_prager.yaml::integration=" in value]
            self.assertTrue(any(value.endswith("::integration=FULL") for value in srm_variant_ids))
            self.assertTrue(any(value.endswith("::integration=B-bar") for value in srm_variant_ids))
            self.assertTrue(any(value.endswith("::integration=SRI") for value in srm_variant_ids))
            self.assertFalse(any(value.endswith("organization_profile.yaml") for value in combo_ids))
            self.assertIn("project_template_load", window.file_actions)

            original_root = window.project_root
            template_path = Path("examples/plane_strain_quad4_bbar.yaml").resolve()
            window.load_project_template_input(template_path)
            QApplication.processEvents()
            self.assertEqual(window.project_root, original_root)
            self.assertIsNone(window.current_input)
            self.assertEqual(window.cfg["mesh"]["integration"], "B-bar")
            self.assertEqual(window.cfg["analysis"]["type"], "static_plane_strain")
            self.assertEqual(len(window.cfg["geometry"]["lines"]), 4)
            self.assertEqual(window.cfg["geometry"]["lines"][0]["source"], "mesh_rectangle_outline")
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertIn("geometry_line", scene_kinds())
            self.assertFalse({"element", "edge", "node"}.intersection(scene_kinds()))
            window._select_tree_panel("analysis")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "analysis")
            tri6_template_path = Path("examples/plane_strain_tri6_patch.yaml").resolve()
            window.load_project_template_input(tri6_template_path)
            QApplication.processEvents()
            self.assertEqual(window.project_root, original_root)
            self.assertEqual(window.cfg["mesh"]["element_type"], "TRI6")
            self.assertEqual(len(window.cfg["geometry"]["lines"]), 3)
            self.assertTrue(all(line.get("source") == "template_mesh_boundary" for line in window.cfg["geometry"]["lines"]))
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertIn("geometry_line", scene_kinds())
            self.assertFalse({"element", "edge", "node"}.intersection(scene_kinds()))
            mesh_step = next(row for row in window.workflow_guidance["steps"] if row["id"] == "mesh")
            self.assertTrue(mesh_step["completed"])
            self.assertFalse(window.input_execution_blockers())
            srm_template_path = Path("examples/slope_srm_drucker_prager.yaml").resolve()
            window.load_project_template_input(f"{srm_template_path}::integration=SRI")
            QApplication.processEvents()
            self.assertEqual(window.project_root, original_root)
            self.assertEqual(window.cfg["mesh"]["integration"], "SRI")
            self.assertEqual(window.cfg["materials"]["colluvium_dp"]["model"], "drucker_prager")
            self.assertEqual([stage["type"] for stage in window.cfg["stages"]], ["srm"])
            self.assertEqual(len(window.cfg["geometry"]["lines"]), 6)
            self.assertEqual(window.cfg["geometry"]["regions"][0]["material"], "colluvium_dp")
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertIn("geometry_line", scene_kinds())
            self.assertIn("geometry_region", scene_kinds())
            self.assertFalse({"element", "edge", "node"}.intersection(scene_kinds()))
            self.assertFalse(window.input_execution_blockers())
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_geometry_view_hides_mesh_and_shape_change_requires_mesh_rebuild(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def scene_kinds(window: object) -> set[str]:
            return {
                str(data.get("kind"))
                for data in (item.data(0) for item in window.scene.items())
                if isinstance(data, dict) and data.get("kind")
            }

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "lines": [
                    {"id": "G1", "purpose": "model", "start": [0.0, 0.0], "end": [10.0, 0.0]},
                    {"id": "G2", "purpose": "model", "start": [10.0, 0.0], "end": [10.0, 2.0]},
                    {"id": "G3", "purpose": "model", "start": [10.0, 2.0], "end": [0.0, 2.0]},
                    {"id": "G4", "purpose": "model", "start": [0.0, 2.0], "end": [0.0, 0.0]},
                ]
            }
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            geometry_kinds = scene_kinds(window)
            self.assertIn("geometry_line", geometry_kinds)
            self.assertFalse({"element", "edge", "node"}.intersection(geometry_kinds))
            self.assertIn("形状のみ表示", window.mesh_summary.text())

            window._select_tree_panel("mesh")
            QApplication.processEvents()
            mesh_kinds = scene_kinds(window)
            self.assertIn("element", mesh_kinds)
            self.assertIn("node", mesh_kinds)

            window._append_geometry_line("model", 2.0, 0.0, 2.0, 2.0)
            window._after_form_change("形状線を追加しました")
            QApplication.processEvents()
            self.assertTrue(window.cfg["mesh"]["requires_rebuild"])
            mesh_step = next(row for row in window.workflow_guidance["steps"] if row["id"] == "mesh")
            self.assertIn("mesh.rebuild_required", mesh_step["missing_paths"])
            self.assertFalse(window.primary_run_button.isEnabled())

            window._select_tree_panel("mesh")
            window.apply_mesh_panel()
            QApplication.processEvents()
            self.assertFalse(bool(window.cfg["mesh"].get("requires_rebuild", False)))
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_mesh_edit_screen_resets_and_rebuilds_mesh_from_shape(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def scene_kinds(window: object) -> set[str]:
            return {
                str(data.get("kind"))
                for data in (item.data(0) for item in window.scene.items())
                if isinstance(data, dict) and data.get("kind")
            }

        def select_first_scene_item(window: object, kind: str) -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == kind:
                    item.setSelected(True)
                    return
            self.fail(f"scene item not found: {kind}")

        def text_button(container: object, label: str) -> QPushButton:
            for button in container.findChildren(QPushButton):
                if button.text() == label:
                    return button
            self.fail(f"button not found: {label}")

        def assert_button_pair(container: object, left_label: str, right_label: str) -> None:
            left = text_button(container, left_label)
            right = text_button(container, right_label)
            self.assertEqual(left.property("uniformPanelButtonRow"), right.property("uniformPanelButtonRow"))
            self.assertEqual(left.property("uniformPanelButtonColumn"), 0)
            self.assertEqual(right.property("uniformPanelButtonColumn"), 1)

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "closure_tolerance": 1.0e-6,
                "lines": [
                    {"id": "G1", "purpose": "model", "start": [0.0, 0.0], "end": [4.0, 0.0]},
                    {"id": "G2", "purpose": "model", "start": [4.0, 0.0], "end": [4.0, 2.0]},
                    {"id": "G3", "purpose": "model", "start": [4.0, 2.0], "end": [0.0, 2.0]},
                    {"id": "G4", "purpose": "model", "start": [0.0, 2.0], "end": [0.0, 0.0]},
                ],
                "regions": [{"id": "R1", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]}],
            }
            cfg["mesh"] = {
                "nodes": {"1": [0.0, 0.0], "2": [4.0, 0.0], "3": [4.0, 2.0], "4": [0.0, 2.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "B-bar"}],
                "node_sets": {"all": ["1", "2", "3", "4"]},
                "element_sets": {"all": ["1"]},
                "mode": "auto_mixed",
                "target_size": 1.0,
                "division_width": 1.0,
            }
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()

            self.assertTrue(window.aux_geometry_settings_group.isVisible())
            self.assertTrue(window.aux_mesh_settings_group.isHidden())
            self.assertFalse(window.model_cad_edit_widget.isVisible())
            self.assertFalse(window.model_mesh_edit_widget.isVisible())
            geometry_equal_inputs = (
                window.model_geometry_point_target,
                window.model_geometry_nudge_step,
                window.model_geometry_point_x,
                window.model_geometry_point_y,
                window.model_geometry_point_dx,
                window.model_geometry_point_dy,
                window.model_cad_length_edit,
                window.model_cad_angle_edit,
            )
            self.assertEqual(
                {widget.parentWidget().property("rightPaneFieldRole") for widget in geometry_equal_inputs},
                {"equal-column"},
            )
            self.assertEqual(window.model_cad_command_edit.parentWidget().property("rightPaneFieldRole"), "full-width")
            cad_undo = text_button(window.aux_geometry_settings_group, "Undo")
            cad_redo = text_button(window.aux_geometry_settings_group, "Redo")
            geometry_buttons = window.aux_geometry_settings_group.findChildren(QPushButton)
            self.assertEqual({button.property("uniformPanelButtonColumns") for button in geometry_buttons}, {2})
            self.assertEqual(len({button.property("uniformPanelButtonWidth") for button in geometry_buttons}), 1)
            assert_button_pair(window.aux_geometry_settings_group, "読込", "入力実行")
            assert_button_pair(window.aux_geometry_settings_group, "座標へ移動", "相対移動")
            assert_button_pair(window.aux_geometry_settings_group, "-X", "+X")
            assert_button_pair(window.aux_geometry_settings_group, "Undo", "Redo")
            initial_line_count = len(window.cfg["geometry"]["lines"])
            window._append_geometry_line("model", 2.0, 0.0, 2.0, 2.0)
            window._after_form_change("cad undo smoke")
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["geometry"]["lines"]), initial_line_count + 1)
            self.assertTrue(window.cfg["mesh"]["requires_rebuild"])
            self.assertTrue(window.aux_edit_history_label.isVisible())
            self.assertIn("直前: cad undo smoke", window.aux_edit_history_label.text())
            cad_undo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["geometry"]["lines"]), initial_line_count)
            self.assertIn("nodes", window.cfg["mesh"])
            self.assertIn("elements", window.cfg["mesh"])
            self.assertFalse(bool(window.cfg["mesh"].get("requires_rebuild", False)))
            self.assertIn("次: cad undo smoke", window.aux_edit_history_label.text())
            cad_redo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["geometry"]["lines"]), initial_line_count + 1)
            self.assertTrue(window.cfg["mesh"]["requires_rebuild"])
            self.assertIn("直前: cad undo smoke", window.aux_edit_history_label.text())
            cad_undo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["geometry"]["lines"]), initial_line_count)

            window._select_tree_panel("mesh")
            QApplication.processEvents()

            self.assertTrue(window.model_mesh_palette_widget.isVisible())
            self.assertTrue(window.aux_mesh_settings_group.isVisible())
            self.assertTrue(window.aux_geometry_settings_group.isHidden())
            self.assertFalse(window.model_cad_palette_widget.isVisible())
            self.assertFalse(window.model_cad_edit_widget.isVisible())
            self.assertFalse(window.model_mesh_edit_widget.isVisible())
            self.assertFalse(window.model_mesh_mode_label.isVisible())
            mesh_undo = text_button(window.aux_mesh_settings_group, "Undo")
            mesh_redo = text_button(window.aux_mesh_settings_group, "Redo")
            mesh_settings_buttons = window.aux_mesh_settings_group.findChildren(QPushButton)
            self.assertEqual({button.property("uniformPanelButtonColumns") for button in mesh_settings_buttons}, {2})
            self.assertEqual(len({button.property("uniformPanelButtonWidth") for button in mesh_settings_buttons}), 1)
            assert_button_pair(window.aux_mesh_settings_group, "再構成", "メッシュ削除")
            assert_button_pair(window.aux_mesh_settings_group, "制御点追加", "選択削除")
            assert_button_pair(window.aux_mesh_settings_group, "品質抽出", "違反修復")
            assert_button_pair(window.aux_mesh_settings_group, "Undo", "Redo")
            self.assertEqual(
                [window.mesh_edit_type.itemText(index) for index in range(window.mesh_edit_type.count())],
                ["QUAD4", "QUAD8", "TRI3", "TRI6"],
            )
            mesh_labels = {
                button.property("meshToolLabel")
                for button in window.model_mesh_palette_widget.findChildren(QPushButton)
                if button.property("meshTool")
            }
            for label in ("選択", "制御点", "局所細分", "分割線", "メッシュ削除", "再構成"):
                self.assertIn(label, mesh_labels)
            self.assertTrue({"element", "node"}.issubset(scene_kinds(window)))

            window.add_mesh_control_point_from_selection()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["mesh"]["control_points"]), 1)
            self.assertIn("mesh_control_point", scene_kinds(window))
            mesh_undo.click()
            QApplication.processEvents()
            self.assertFalse(window.cfg["mesh"].get("control_points"))
            mesh_redo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["mesh"]["control_points"]), 1)
            select_first_scene_item(window, "mesh_control_point")
            window.delete_selected_mesh_edit_items()
            QApplication.processEvents()
            self.assertEqual(window.cfg["mesh"].get("control_points"), [])

            window._append_geometry_line("model", 2.0, 0.0, 2.0, 2.0)
            window._after_form_change("形状線を追加しました")
            QApplication.processEvents()
            self.assertTrue(window.cfg["mesh"]["requires_rebuild"])
            self.assertNotIn("nodes", window.cfg["mesh"])
            self.assertNotIn("elements", window.cfg["mesh"])
            self.assertIn("メッシュ未生成", window.mesh_summary.text())
            self.assertFalse({"element", "node"}.intersection(scene_kinds(window)))
            self.assertIn("geometry_line", scene_kinds(window))

            window.mesh_edit_type.setCurrentText("QUAD8")
            window.mesh_edit_target_size.setText("0.5")
            window.mesh_edit_nx.setText("8")
            window.mesh_edit_ny.setText("4")
            window.apply_mesh_edit_generation_settings()
            QApplication.processEvents()
            self.assertEqual(window.cfg["mesh"]["element_type"], "QUAD8")
            self.assertEqual(window.cfg["mesh"]["requested_element_type"], "QUAD8")
            self.assertEqual(window.cfg["mesh"]["target_size"], 0.5)
            self.assertEqual(window.cfg["mesh"]["nx"], 8)
            self.assertEqual(window.cfg["mesh"]["ny"], 4)
            self.assertTrue(window.cfg["mesh"]["requires_rebuild"])

            window.rebuild_mesh_from_shape()
            QApplication.processEvents()
            self.assertFalse(bool(window.cfg["mesh"].get("requires_rebuild", False)))
            self.assertIn("nodes", window.cfg["mesh"])
            self.assertIn("elements", window.cfg["mesh"])
            self.assertGreater(len(window.cfg["mesh"]["elements"]), 0)
            self.assertEqual({element["type"] for element in window.cfg["mesh"]["elements"]}, {"QUAD8"})
            self.assertTrue({"element", "node"}.issubset(scene_kinds(window)))

            window.reset_mesh_for_rebuild()
            QApplication.processEvents()
            self.assertTrue(window.cfg["mesh"]["requires_rebuild"])
            self.assertNotIn("nodes", window.cfg["mesh"])
            self.assertNotIn("elements", window.cfg["mesh"])
            self.assertFalse({"element", "node"}.intersection(scene_kinds(window)))

            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
            def rule_buttons(panel: str, rule: str) -> list[QPushButton]:
                right_pane_groups = {
                    "boundary_conditions": window.aux_boundary_action_group,
                    "loads": window.aux_load_action_group,
                }
                container = right_pane_groups.get(panel, window.panel_pages[panel])
                return [
                    button
                    for button in container.findChildren(QPushButton)
                    if str(button.property("operationRule") or "") == rule
                ]

            for panel in ("materials", "boundary_conditions", "loads"):
                window._select_tree_panel(panel)
                QApplication.processEvents()
                self.assertTrue(window.inline_detail_sheet_group.isVisible())
                self.assertEqual(window.inline_detail_stack.currentWidget(), window.panel_pages[panel])
                self.assertEqual(window.tabs.currentWidget(), window.model_workspace_page)
                self.assertTrue(window.view_toolbar_selection_widget.isHidden())
                self.assertTrue(window.selection_builder_toolbar_widget.isHidden())
                self.assertFalse(hasattr(window, "aux_open_detail_button"))
                self.assertFalse(hasattr(window, "aux_model_check_button"))
                self.assertGreaterEqual(window.inline_detail_sheet_group.minimumHeight(), 220)
                self.assertLessEqual(window.inline_detail_sheet_group.maximumHeight(), 320)
                if panel == "materials":
                    self.assertTrue(window.aux_material_action_group.isVisible())
                    self.assertTrue(window.aux_boundary_action_group.isHidden())
                    self.assertTrue(window.aux_load_action_group.isHidden())
                    self.assertFalse(any(button.text() == "Undo" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(any(button.text() == "Redo" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertIsNotNone(text_button(window.aux_material_action_group, "Undo"))
                    self.assertIsNotNone(text_button(window.aux_material_action_group, "Redo"))
                    self.assertIsNotNone(text_button(window.aux_material_action_group, "MC"))
                    self.assertIsNotNone(text_button(window.aux_material_action_group, "DP"))
                    material_buttons = window.aux_material_action_group.findChildren(QPushButton)
                    self.assertGreaterEqual(len(material_buttons), 10)
                    self.assertEqual({button.property("uniformPanelButtonColumns") for button in material_buttons}, {2})
                    self.assertEqual(len({button.property("uniformPanelButtonWidth") for button in material_buttons}), 1)
                    assert_button_pair(window.aux_material_action_group, "追加", "削除")
                    assert_button_pair(window.aux_material_action_group, "MC", "DP")
                    assert_button_pair(window.aux_material_action_group, "Undo", "Redo")
                    self.assertFalse(hasattr(window, "aux_model_check_button"))
                elif panel == "boundary_conditions":
                    self.assertTrue(window.aux_boundary_action_group.isVisible())
                    self.assertTrue(window.aux_material_action_group.isHidden())
                    self.assertTrue(window.aux_load_action_group.isHidden())
                    self.assertFalse(any(button.text() == "Undo" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(any(button.text() == "Redo" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(any(button.text() == "境界条件YAMLを反映" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(window.boundary_condition_tree.isVisible())
                    self.assertFalse(window.boundary_conditions_editor.isVisible())
                    boundary_buttons = window.aux_boundary_action_group.findChildren(QPushButton)
                    self.assertEqual({button.property("uniformPanelButtonColumns") for button in boundary_buttons}, {2})
                    self.assertEqual(len({button.property("uniformPanelButtonWidth") for button in boundary_buttons}), 1)
                    assert_button_pair(window.aux_boundary_action_group, "水平ローラ", "鉛直ローラ")
                    assert_button_pair(window.aux_boundary_action_group, "削除", "表を反映")
                    assert_button_pair(window.aux_boundary_action_group, "選択節点→支点/変位", "選択辺/要素→水理境界")
                    assert_button_pair(window.aux_boundary_action_group, "Undo", "Redo")
                else:
                    self.assertTrue(window.aux_load_action_group.isVisible())
                    self.assertTrue(window.aux_material_action_group.isHidden())
                    self.assertTrue(window.aux_boundary_action_group.isHidden())
                    self.assertFalse(any(button.text() == "Undo" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(any(button.text() == "Redo" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(any(button.text() == "荷重YAMLを反映" for button in window.panel_pages[panel].findChildren(QPushButton)))
                    self.assertFalse(window.loads_editor.isVisible())
                    load_buttons = window.aux_load_action_group.findChildren(QPushButton)
                    self.assertEqual({button.property("uniformPanelButtonColumns") for button in load_buttons}, {2})
                    self.assertEqual(len({button.property("uniformPanelButtonWidth") for button in load_buttons}), 1)
                    assert_button_pair(window.aux_load_action_group, "静的ケース追加", "地震ケース追加")
                    assert_button_pair(window.aux_load_action_group, "ケース削除", "荷重削除")
                    assert_button_pair(window.aux_load_action_group, "選択辺/要素→面荷重", "選択辺/要素→偏分布面荷重")
                    assert_button_pair(window.aux_load_action_group, "Undo", "Redo")
                    load_batch_widgets = (
                        window.load_batch_scope,
                        window.load_case_selector,
                        window.load_body_material,
                        window.load_surface_distribution,
                        window.load_body_bx,
                        window.load_body_by,
                        window.load_fx,
                        window.load_fy,
                        window.load_tx,
                        window.load_ty,
                        window.load_tx_end,
                        window.load_ty_end,
                        window.load_seismic_kh,
                        window.load_seismic_kv,
                        window.load_seismic_direction,
                    )
                    self.assertEqual({widget.property("loadBatchGridColumns") for widget in load_batch_widgets}, {8})

                    def assert_load_grid_pair(left: object, right: object) -> None:
                        self.assertEqual(left.property("loadBatchGridRow"), right.property("loadBatchGridRow"))
                        self.assertEqual(
                            int(left.property("loadBatchGridPairColumn")) + 1,
                            int(right.property("loadBatchGridPairColumn")),
                        )

                    assert_load_grid_pair(window.load_body_bx, window.load_body_by)
                    assert_load_grid_pair(window.load_fx, window.load_fy)
                    assert_load_grid_pair(window.load_tx, window.load_ty)
                    assert_load_grid_pair(window.load_tx_end, window.load_ty_end)
                    assert_load_grid_pair(window.load_seismic_kh, window.load_seismic_kv)
            self.assertEqual(window.material_table.horizontalHeaderItem(0).text(), "材料名")
            self.assertEqual(window.material_table.horizontalHeaderItem(1).text(), "モデル")
            self.assertEqual(window.boundary_table.horizontalHeaderItem(0).text(), "対象(節点/セット)")
            self.assertEqual(window.load_case_table.horizontalHeaderItem(0).text(), "ケース")
            self.assertEqual(window.load_table.horizontalHeaderItem(0).text(), "種別")

            def workflow_button_for_panel(panel: str) -> QPushButton:
                steps = [step for step in window.workflow_guidance.get("steps", []) if isinstance(step, dict)]
                for index, step in enumerate(steps):
                    if step.get("panel") == panel:
                        return window.workflow_ribbon_step_buttons[index]
                self.fail(f"workflow button not found: {panel}")

            for panel, expected_widget in (
                ("analysis", window.panel_stack),
                ("geometry", window.model_workspace_page),
                ("mesh", window.model_workspace_page),
                ("materials", window.model_workspace_page),
                ("boundary_conditions", window.model_workspace_page),
                ("loads", window.model_workspace_page),
                ("stages", window.model_workspace_page),
                ("model_check", window.panel_stack),
                ("solver", window.panel_stack),
            ):
                window._select_tree_panel(panel)
                QApplication.processEvents()
                self.assertEqual(window.current_panel_key, panel)
                self.assertIs(window.tabs.currentWidget(), expected_widget)
                workflow_button_for_panel(panel).click()
                QApplication.processEvents()
                self.assertEqual(window.current_panel_key, panel)
                self.assertIs(window.tabs.currentWidget(), expected_widget)

            window._select_tree_panel("stages")
            QApplication.processEvents()
            stage_center_width = window.workspace_splitter.sizes()[1]
            for panel in ("model_check", "solver"):
                window._select_tree_panel(panel)
                QApplication.processEvents()
                self.assertEqual(window.workspace_splitter.sizes()[1], stage_center_width)

            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertFalse(window.inline_detail_sheet_group.isVisible())

            window._select_tree_panel("materials")
            QApplication.processEvents()
            window.tabs.setCurrentWidget(window.check_table)
            window.open_current_detail_panel()
            QApplication.processEvents()
            self.assertEqual(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertEqual(window.inline_detail_stack.currentWidget(), window.panel_pages["materials"])

            window.scene.clearSelection()
            window._select_tree_panel("boundary_conditions")
            QApplication.processEvents()
            boundary_action = rule_buttons("boundary_conditions", "boundary_nodes_edges")[0]
            boundary_hydro = rule_buttons("boundary_conditions", "boundary_hydro")[0]
            boundary_mpc = rule_buttons("boundary_conditions", "boundary_mpc")[0]
            self.assertFalse(boundary_action.isEnabled())
            self.assertFalse(boundary_hydro.isEnabled())
            self.assertFalse(boundary_mpc.isEnabled())
            select_first_scene_item(window, "node")
            QApplication.processEvents()
            self.assertTrue(boundary_action.isEnabled())
            self.assertFalse(boundary_mpc.isEnabled())
            window.scene.clearSelection()
            select_first_scene_item(window, "element")
            QApplication.processEvents()
            self.assertFalse(boundary_action.isEnabled())

            window.scene.clearSelection()
            window._select_tree_panel("loads")
            QApplication.processEvents()
            load_nodes = rule_buttons("loads", "load_nodes")[0]
            load_body = rule_buttons("loads", "load_body")[0]
            load_surface = rule_buttons("loads", "load_edges_or_elements")[0]
            self.assertFalse(load_nodes.isEnabled())
            self.assertFalse(load_body.isEnabled())
            self.assertFalse(load_surface.isEnabled())
            select_first_scene_item(window, "node")
            QApplication.processEvents()
            self.assertTrue(load_nodes.isEnabled())
            self.assertFalse(load_body.isEnabled())
            window.scene.clearSelection()
            select_first_scene_item(window, "element")
            QApplication.processEvents()
            self.assertFalse(load_nodes.isEnabled())
            self.assertTrue(load_body.isEnabled())
            self.assertTrue(load_surface.isEnabled())
            window.scene.clearSelection()
            index = window.load_body_material.findData("soil")
            if index >= 0:
                window.load_body_material.setCurrentIndex(index)
            window._refresh_operation_button_enabled_states()
            self.assertTrue(load_body.isEnabled())
            window._select_tree_panel("boundary_conditions")
            QApplication.processEvents()
            self.assertFalse(load_body.isEnabled())

            window._select_tree_panel("materials")
            QApplication.processEvents()
            material_undo = text_button(window.aux_material_action_group, "Undo")
            material_redo = text_button(window.aux_material_action_group, "Redo")
            window.cfg.setdefault("materials", {})["undo_probe"] = {"model": "elastic", "E": 123.0, "nu": 0.25}
            window._after_form_change("material undo smoke")
            self.assertIn("直前: material undo smoke", window.aux_edit_history_label.text())
            material_undo.click()
            QApplication.processEvents()
            self.assertNotIn("undo_probe", window.cfg.get("materials", {}))
            self.assertIn("次: material undo smoke", window.aux_edit_history_label.text())
            material_redo.click()
            QApplication.processEvents()
            self.assertIn("undo_probe", window.cfg.get("materials", {}))
            self.assertIn("直前: material undo smoke", window.aux_edit_history_label.text())

            window._select_tree_panel("boundary_conditions")
            QApplication.processEvents()
            boundary_undo = text_button(window.aux_boundary_action_group, "Undo")
            boundary_redo = text_button(window.aux_boundary_action_group, "Redo")
            if not isinstance(window.cfg.get("boundary_conditions"), list):
                window.cfg["boundary_conditions"] = []
            boundary_count = len(window.cfg["boundary_conditions"])
            window.cfg["boundary_conditions"].append({"node": "1", "ux": 0.0})
            window._after_form_change("boundary undo smoke")
            boundary_undo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg.get("boundary_conditions", [])), boundary_count)
            self.assertIn("次: boundary undo smoke", window.aux_edit_history_label.text())
            boundary_redo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg.get("boundary_conditions", [])), boundary_count + 1)
            self.assertIn("直前: boundary undo smoke", window.aux_edit_history_label.text())

            window._select_tree_panel("loads")
            QApplication.processEvents()
            loads_undo = text_button(window.aux_load_action_group, "Undo")
            loads_redo = text_button(window.aux_load_action_group, "Redo")
            if not isinstance(window.cfg.get("loads"), list):
                window.cfg["loads"] = []
            loads_count = len(window.cfg["loads"])
            window.cfg["loads"].append({"type": "node", "node": "2", "fy": -1.0})
            window._after_form_change("loads undo smoke")
            loads_undo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg.get("loads", [])), loads_count)
            self.assertIn("次: loads undo smoke", window.aux_edit_history_label.text())
            loads_redo.click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg.get("loads", [])), loads_count + 1)
            self.assertIn("直前: loads undo smoke", window.aux_edit_history_label.text())

            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
            for index in range(12):
                window.cfg.setdefault("analysis", {})["undo_probe"] = index
                window._after_form_change(f"history limit smoke {index}")
            self.assertEqual(len(window.edit_undo_stack), 10)
            self.assertEqual(len(window.edit_undo_labels), 10)
            self.assertIn("history limit smoke 11", window.aux_edit_history_label.text())

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_region_meshes_keep_individual_density_and_delete_by_shape(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "regions": [
                    {"id": "R1", "points": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]},
                    {"id": "R2", "points": [[2.0, 0.0], [4.0, 0.0], [4.0, 1.0], [2.0, 1.0]]},
                ]
            }
            cfg["mesh"] = {
                "mode": "auto_mixed",
                "target_size": 1.0,
                "division_width": 1.0,
                "element_type": "QUAD4",
                "region_settings": {
                    "R2": {"target_size": 0.5, "division_width": 0.5, "element_type": "TRI3", "nx": 4, "ny": 2}
                },
            }
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("mesh")
            QApplication.processEvents()
            window.rebuild_mesh_from_shape()
            QApplication.processEvents()

            mesh = window.cfg["mesh"]
            self.assertEqual(mesh["mode"], "per_region_mapped")
            self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4", "TRI3"})
            self.assertGreater(len(mesh["element_sets"]["region_R2"]), len(mesh["element_sets"]["region_R1"]))
            self.assertGreater(mesh["mesh_quality"]["per_region_mapped_shared_node_count"], 0)
            shape_by_id = {shape["region_id"]: shape for shape in mesh["shape_meshes"]}
            self.assertEqual(shape_by_id["R1"]["element_type"], "QUAD4")
            self.assertEqual(shape_by_id["R2"]["element_type"], "TRI3")
            self.assertGreater(shape_by_id["R1"]["transition_grading"]["right"], 1.0)
            bottom_r1_x = sorted(
                float(xy[0])
                for xy in mesh["nodes"].values()
                if abs(float(xy[1])) <= 1.0e-9 and -1.0e-9 <= float(xy[0]) <= 2.0 + 1.0e-9
            )
            self.assertGreater(bottom_r1_x[1] - bottom_r1_x[0], bottom_r1_x[-1] - bottom_r1_x[-2])

            r2_element = mesh["element_sets"]["region_R2"][0]
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "element" and str(data.get("id")) == str(r2_element):
                    item.setSelected(True)
                    break
            else:
                self.fail("R2 element scene item not found")
            window.delete_selected_mesh_edit_items()
            QApplication.processEvents()
            partial_mesh = window.cfg["mesh"]
            self.assertTrue(partial_mesh["requires_rebuild"])
            self.assertTrue(partial_mesh["partial_rebuild_required"])
            self.assertIn("R2", partial_mesh["deleted_region_meshes"])
            self.assertIn("nodes", partial_mesh)
            self.assertIn("elements", partial_mesh)
            self.assertEqual({element["type"] for element in partial_mesh["elements"]}, {"QUAD4"})
            partial_shapes = {shape["region_id"]: shape for shape in partial_mesh["shape_meshes"]}
            self.assertEqual(partial_shapes["R2"]["status"], "deleted_requires_rebuild")
            self.assertGreater(len(partial_shapes["R1"]["element_ids"]), 0)

            window.rebuild_mesh_from_shape()
            QApplication.processEvents()
            rebuilt = window.cfg["mesh"]
            self.assertFalse(bool(rebuilt.get("requires_rebuild", False)))
            self.assertEqual({element["type"] for element in rebuilt["elements"]}, {"QUAD4", "TRI3"})
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_cad_palette_sets_selected_region_grid_and_context_menu_is_task_scoped(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def select_region(window: object, region_id: str) -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "geometry_region" and str(data.get("id")) == region_id:
                    item.setSelected(True)
                    return
            self.fail(f"geometry region not found: {region_id}")

        def visible_labels(specs: list[tuple[str, object, str]]) -> set[str]:
            return {label for label, _callback, _tip in specs if label != "-"}

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "regions": [
                    {"id": "R1", "points": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]},
                    {"id": "R2", "points": [[2.0, 0.0], [4.0, 0.0], [4.0, 1.0], [2.0, 1.0]]},
                ]
            }
            cfg["mesh"] = {"mode": "auto_mixed", "target_size": 1.0, "division_width": 1.0, "element_type": "QUAD4", "nx": 2, "ny": 1}
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()

            cad_labels = {
                button.property("cadToolLabel")
                for button in window.model_cad_palette_widget.findChildren(QPushButton)
                if button.property("cadTool")
            }
            self.assertIn("格子条件", cad_labels)
            geometry_context = visible_labels(window._context_menu_action_specs("view"))
            self.assertIn("選択形状の格子条件", geometry_context)
            self.assertIn("領域化", geometry_context)
            self.assertNotIn("解析実行", geometry_context)
            self.assertNotIn("モデルチェック", geometry_context)

            select_region(window, "R2")
            window.apply_selected_geometry_grid_settings(text="TRI3 0.5 4 2")
            QApplication.processEvents()
            mesh = window.cfg["mesh"]
            self.assertNotIn("nodes", mesh)
            self.assertNotIn("elements", mesh)
            self.assertTrue(mesh["requires_rebuild"])
            self.assertTrue(mesh["partial_rebuild_required"])
            self.assertEqual(mesh["region_rebuild_required"], ["R2"])
            self.assertEqual(mesh["region_settings"]["R2"]["element_type"], "TRI3")
            self.assertEqual(mesh["region_settings"]["R2"]["nx"], 4)
            self.assertEqual(mesh["region_settings"]["region_2"]["target_size"], 0.5)
            self.assertTrue(window.model_cad_palette_widget.isVisible())
            self.assertTrue(window.model_mesh_palette_widget.isHidden())

            window._select_tree_panel("mesh")
            QApplication.processEvents()
            mesh_context = visible_labels(window._context_menu_action_specs("view"))
            self.assertIn("選択形状のメッシュ条件反映", mesh_context)
            self.assertIn("メッシュ再構成", mesh_context)
            self.assertNotIn("解析実行", mesh_context)
            window.rebuild_mesh_from_shape()
            QApplication.processEvents()
            rebuilt = window.cfg["mesh"]
            shape_by_id = {shape["region_id"]: shape for shape in rebuilt["shape_meshes"]}
            self.assertEqual(shape_by_id["R1"]["element_type"], "QUAD4")
            self.assertEqual(shape_by_id["R2"]["element_type"], "TRI3")
            self.assertEqual(rebuilt["region_settings"]["R2"]["ny"], 2)

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_region_material_assignment_colors_and_mesh_materials(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QComboBox, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def select_region(window: object, region_id: str) -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "geometry_region" and str(data.get("id")) == region_id:
                    item.setSelected(True)
                    return
            self.fail(f"geometry region not found: {region_id}")

        def region_item(window: object, region_id: str) -> object:
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "geometry_region" and str(data.get("id")) == region_id:
                    return item
            self.fail(f"geometry region not found: {region_id}")

        def context_labels(window: object) -> set[str]:
            return {label for label, _callback, _tip in window._context_menu_action_specs("view") if label != "-"}

        def context_callback_names(window: object) -> set[str]:
            return {
                str(getattr(callback, "__name__", ""))
                for label, callback, _tip in window._context_menu_action_specs("view")
                if label != "-" and callback is not None
            }

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "regions": [
                    {"id": "R1", "points": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]},
                    {"id": "R2", "points": [[2.0, 0.0], [4.0, 0.0], [4.0, 1.0], [2.0, 1.0]]},
                ]
            }
            cfg["mesh"] = {"mode": "auto_mixed", "target_size": 1.0, "division_width": 1.0, "element_type": "QUAD4", "nx": 2, "ny": 1}
            cfg["materials"] = {
                "soil": {"model": "elastic", "E": 50000.0, "nu": 0.3, "gamma": 18.0},
                "clay": {"model": "mohr_coulomb", "E": 25000.0, "nu": 0.35, "cohesion": 20.0, "friction_angle": 24.0, "gamma": 17.0},
            }
            window._load_cfg(cfg, keep_yaml=False)
            material_issue_text = "\n".join(window.workflow_step_issue_lines("materials", "materials"))
            self.assertIn("未割当材料: R1、R2", material_issue_text)
            window._select_tree_panel("geometry")
            QApplication.processEvents()

            self.assertTrue(any(blocker.get("source") == "material_assignment" for blocker in window.input_execution_blockers()))
            labels = context_labels(window)
            self.assertNotIn("材料: soil", labels)
            self.assertNotIn("材料: clay", labels)
            self.assertNotIn("新規材料を詳細で定義", labels)
            palette_labels = {label for label, _callback, _tip in window._context_menu_action_specs("cad_palette") if label != "-"}
            self.assertNotIn("新規材料を詳細で定義", palette_labels)
            self.assertEqual(window.model_cad_palette_widget.property("workContextMenu"), "cad_palette")
            self.assertEqual(window.model_cad_palette_widget.contextMenuPolicy(), Qt.ContextMenuPolicy.CustomContextMenu)
            cad_buttons = [
                button
                for button in window.model_cad_palette_widget.findChildren(QPushButton)
                if button.property("cadTool")
            ]
            self.assertTrue(cad_buttons)
            self.assertTrue(all(button.property("workContextMenu") == "cad_palette" for button in cad_buttons))

            window._select_tree_panel("mesh")
            QApplication.processEvents()
            labels = context_labels(window)
            self.assertIn("メッシュ再構成", labels)
            self.assertNotIn("材料: soil", labels)
            self.assertNotIn("材料: clay", labels)
            self.assertNotIn("新規材料を詳細で定義", labels)
            palette_labels = {label for label, _callback, _tip in window._context_menu_action_specs("mesh_palette") if label != "-"}
            self.assertNotIn("新規材料を詳細で定義", palette_labels)

            window._select_tree_panel("boundary_conditions")
            QApplication.processEvents()
            boundary_callbacks = context_callback_names(window)
            self.assertIn("add_selected_hydro_boundary_condition", boundary_callbacks)
            self.assertIn("add_selected_mpc_constraints", boundary_callbacks)
            self.assertIn("apply_boundary_conditions_panel", boundary_callbacks)
            self.assertNotIn("register_selected_nodes_as_set", boundary_callbacks)
            self.assertNotIn("reset_to_scene", boundary_callbacks)
            self.assertNotIn("apply_materials_panel", boundary_callbacks)
            self.assertNotIn("add_selected_nodal_load_condition", boundary_callbacks)

            window._select_tree_panel("loads")
            QApplication.processEvents()
            load_callbacks = context_callback_names(window)
            self.assertIn("add_selected_nodal_load_condition", load_callbacks)
            self.assertIn("add_selected_body_load_condition", load_callbacks)
            self.assertIn("apply_loads_panel", load_callbacks)
            self.assertNotIn("reset_to_scene", load_callbacks)
            self.assertNotIn("apply_materials_panel", load_callbacks)
            self.assertNotIn("add_selected_boundary_condition", load_callbacks)

            window._select_tree_panel("materials")
            QApplication.processEvents()
            labels = context_labels(window)
            self.assertIn("選択", labels)
            self.assertIn("材料: soil", labels)
            self.assertIn("材料: clay", labels)
            self.assertIn("材料割当解除", labels)
            self.assertIn("新規材料を詳細で定義", labels)
            self.assertIn("材料を反映", labels)
            material_callbacks = context_callback_names(window)
            self.assertIn("apply_materials_panel", material_callbacks)
            self.assertNotIn("reset_to_scene", material_callbacks)
            self.assertNotIn("add_selected_boundary_condition", material_callbacks)
            self.assertNotIn("add_selected_nodal_load_condition", material_callbacks)
            window.activate_material_assignment_selection()
            self.assertEqual(window.draw_mode, "select")
            self.assertFalse(window._operation_rule_state("material_assign")[0])
            self.assertIn("現在の作業:", window.aux_current_task_label.text())
            self.assertIn("右クリック:", window.aux_context_menu_status_label.text())

            select_region(window, "R1")
            QApplication.processEvents()
            self.assertTrue(window._operation_rule_state("material_assign")[0])
            self.assertIn("形状R1", window.aux_current_selection_label.text())
            self.assertIn("右クリック: 可", window.aux_context_menu_status_label.text())
            selected_before = {
                str(item.data(0).get("id"))
                for item in window.scene.selectedItems()
                if isinstance(item.data(0), dict) and item.data(0).get("kind") == "geometry_region"
            }
            context_calls: list[str] = []
            original_context_menu = window._show_command_context_menu
            window._show_command_context_menu = lambda _global_pos, context: context_calls.append(context)
            try:
                right_click_pos = window.view.mapFromScene(region_item(window, "R1").sceneBoundingRect().center())
                window.view.mousePressEvent(
                    type(
                        "FakeRightPress",
                        (),
                        {
                            "button": lambda self: Qt.MouseButton.RightButton,
                            "pos": lambda self, p=right_click_pos: p,
                            "accept": lambda self: None,
                        },
                    )()
                )
                window.view.mouseReleaseEvent(
                    type(
                        "FakeRightRelease",
                        (),
                        {
                            "button": lambda self: Qt.MouseButton.RightButton,
                            "pos": lambda self, p=right_click_pos: p,
                            "accept": lambda self: None,
                        },
                    )()
                )
                self.assertEqual(context_calls, [])
                window.view.contextMenuEvent(
                    type(
                        "FakeContextMenuEvent",
                        (),
                        {
                            "pos": lambda self, p=right_click_pos: p,
                            "globalPos": lambda self, p=window.view.viewport().mapToGlobal(right_click_pos): p,
                            "accept": lambda self: None,
                        },
                    )()
                )
            finally:
                window._show_command_context_menu = original_context_menu
            selected_after = {
                str(item.data(0).get("id"))
                for item in window.scene.selectedItems()
                if isinstance(item.data(0), dict) and item.data(0).get("kind") == "geometry_region"
            }
            self.assertEqual(context_calls, ["view"])
            self.assertEqual(selected_after, selected_before)
            window.assign_material_to_selected_geometry_regions("soil")
            QApplication.processEvents()
            select_region(window, "R2")
            window.assign_material_to_selected_geometry_regions("clay")
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["regions"][0]["material"], "soil")
            self.assertEqual(window.cfg["geometry"]["regions"][1]["material"], "clay")
            self.assertEqual(window.cfg["mesh"]["region_settings"]["R2"]["material"], "clay")
            self.assertFalse(bool(window.cfg["mesh"].get("requires_rebuild", False)))
            self.assertFalse(bool(window.cfg["mesh"].get("partial_rebuild_required", False)))
            self.assertFalse(any(blocker.get("source") == "material_assignment" for blocker in window.input_execution_blockers()))
            self.assertEqual(str(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole)), "material_assignment:region:R2")

            root = window.tree.topLevelItem(0)
            soil_item = window._find_tree_panel_item(root, "material_assignment:material:soil")
            self.assertIsNotNone(soil_item)
            window.tree.setCurrentItem(soil_item)
            QApplication.processEvents()
            selected_regions = {
                str(item.data(0).get("id"))
                for item in window.scene.selectedItems()
                if isinstance(item.data(0), dict) and item.data(0).get("kind") == "geometry_region"
            }
            self.assertIn("R1", selected_regions)
            self.assertNotIn("R2", selected_regions)

            r1_color = region_item(window, "R1").brush().color().name().lower()
            r2_color = region_item(window, "R2").brush().color().name().lower()
            self.assertNotEqual(r1_color, r2_color)

            window.rebuild_mesh_from_shape()
            QApplication.processEvents()
            by_region = {}
            for element in window.cfg["mesh"]["elements"]:
                by_region.setdefault(element.get("region_id"), set()).add(element.get("material"))
            self.assertEqual(by_region["R1"], {"soil"})
            self.assertEqual(by_region["R2"], {"clay"})

            element_count_after_rebuild = len(window.cfg["mesh"]["elements"])
            select_region(window, "R2")
            window.assign_material_to_selected_geometry_regions("soil")
            QApplication.processEvents()
            self.assertFalse(bool(window.cfg["mesh"].get("requires_rebuild", False)))
            self.assertFalse(bool(window.cfg["mesh"].get("partial_rebuild_required", False)))
            self.assertEqual(len(window.cfg["mesh"]["elements"]), element_count_after_rebuild)
            by_region = {}
            for element in window.cfg["mesh"]["elements"]:
                by_region.setdefault(element.get("region_id"), set()).add(element.get("material"))
            self.assertEqual(by_region["R1"], {"soil"})
            self.assertEqual(by_region["R2"], {"soil"})

            select_region(window, "R2")
            window.clear_material_from_selected_geometry_regions()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["mesh"]["elements"]), element_count_after_rebuild)
            self.assertNotIn("material", window.cfg["geometry"]["regions"][1])
            self.assertNotIn("material", window.cfg["mesh"]["region_settings"]["R2"])
            by_region = {}
            for element in window.cfg["mesh"]["elements"]:
                by_region.setdefault(element.get("region_id"), set()).add(element.get("material"))
            self.assertEqual(by_region["R1"], {"soil"})
            self.assertEqual(by_region["R2"], {""})
            self.assertTrue(any(blocker.get("source") == "material_assignment" for blocker in window.input_execution_blockers()))
            self.assertIn("未割当材料: R2", "\n".join(window.workflow_step_issue_lines("materials", "materials")))

            select_region(window, "R2")
            window.assign_material_to_selected_geometry_regions("soil")
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["mesh"]["elements"]), element_count_after_rebuild)
            self.assertFalse(any(blocker.get("source") == "material_assignment" for blocker in window.input_execution_blockers()))

            window.add_material_row(name="steel", model="von_mises", E=200000.0, nu=0.29, yield_stress=250.0)
            model_widget = window.material_table.cellWidget(window.material_table.rowCount() - 1, 1)
            self.assertIsInstance(model_widget, QComboBox)
            self.assertGreaterEqual(model_widget.minimumWidth(), 220)
            self.assertGreaterEqual(model_widget.view().minimumWidth(), 340)
            self.assertNotIn("Linear elastic", model_widget.currentText())
            self.assertIn("塑性", model_widget.currentText())
            self.assertEqual(window._table_text(window.material_table, window.material_table.rowCount() - 1, 1), "von_mises")

            select_region(window, "R2")
            window.open_material_detail_for_new()
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "materials")
            self.assertEqual(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertEqual(window.inline_detail_stack.currentWidget(), window.panel_pages["materials"])
            self.assertEqual(window.material_library_name.text(), "mat_R2")

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_stage_context_menu_adds_selected_elements_to_current_stage(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def select_element(window: object, element_id: str) -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "element" and str(data.get("id")) == element_id:
                    item.setSelected(True)
                    return
            self.fail(f"element not found: {element_id}")

        def visible_labels(window: object) -> set[str]:
            return {label for label, _callback, _tip in window._context_menu_action_specs("view") if label != "-"}

        def callback_names(window: object) -> set[str]:
            return {
                str(getattr(callback, "__name__", ""))
                for label, callback, _tip in window._context_menu_action_specs("view")
                if label != "-" and callback is not None
            }

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["stages"] = [{"name": "Stage-1", "type": "static"}]
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("stages")
            QApplication.processEvents()

            labels = visible_labels(window)
            self.assertIn("選択要素をステージsetへ", labels)
            self.assertIn("選択要素を無効化", labels)
            self.assertIn("選択要素を再有効化", labels)
            self.assertIn("選択節点/辺へ境界変位", labels)
            self.assertIn("選択辺/要素へ分布荷重", labels)
            self.assertNotIn("解析実行", labels)
            self.assertNotIn("モデルチェック", labels)
            callbacks = callback_names(window)
            self.assertIn("add_selected_elements_to_stage_death", callbacks)
            self.assertIn("add_selected_elements_to_stage_birth", callbacks)
            self.assertFalse(window._operation_rule_state("stage_elements")[0])

            select_element(window, "1")
            QApplication.processEvents()
            self.assertTrue(window._operation_rule_state("stage_elements")[0])
            self.assertTrue(window._operation_rule_state("stage_edges")[0])
            self.assertIn("右クリック: 可", window._right_click_context_text("stages"))

            window.add_selected_elements_to_stage_death()
            QApplication.processEvents()
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["type"], "death")
            self.assertEqual(stage["elements"], ["1"])
            self.assertEqual(stage["construction_events"][0]["action"], "death")
            self.assertEqual(stage["construction_events"][0]["elements"], ["1"])
            self.assertIn("無効化", window.stage_change_table.item(0, 1).text())

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_cad_selection_moves_shared_vertex_and_registers_closed_lines_as_region(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def scene_point(window: object, x: float, y: float) -> QPointF:
            return QPointF(x * window.preview_scale + window.preview_ox, window.preview_oy - y * window.preview_scale)

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "closure_tolerance": 1.0e-6,
                "lines": [
                    {"id": "L1", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]},
                    {"id": "L2", "purpose": "model", "start": [1.0, 0.0], "end": [1.0, 1.0]},
                    {"id": "L3", "purpose": "model", "start": [1.0, 1.0], "end": [0.0, 1.0]},
                    {"id": "L4", "purpose": "model", "start": [0.0, 1.0], "end": [0.0, 0.0]},
                ],
                "regions": [],
            }
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            window.snap_enabled.setChecked(False)
            QApplication.processEvents()

            endpoint_data = window.nearest_geometry_endpoint_at_scene_point(scene_point(window, 1.0, 0.0))
            self.assertEqual(endpoint_data["id"], "L1")
            self.assertEqual(endpoint_data["endpoint"], "end")
            window.begin_endpoint_drag(endpoint_data)
            window.update_endpoint_drag(scene_point(window, 1.2, 0.25), final=False)
            window.update_endpoint_drag(scene_point(window, 1.2, 0.25), final=True)
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["end"], [1.2, 0.25])
            self.assertEqual(window.cfg["geometry"]["lines"][1]["start"], [1.2, 0.25])

            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "geometry_line":
                    item.setSelected(True)
            window.register_closed_lines_as_region()
            QApplication.processEvents()
            regions = window.cfg["geometry"]["regions"]
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0]["source"], "cad_closed_lines")
            self.assertEqual(set(regions[0]["source_lines"]), {"L1", "L2", "L3", "L4"})
            self.assertEqual(len(regions[0]["points"]), 4)
            self.assertTrue(any(abs(point[0] - 1.2) < 1.0e-9 and abs(point[1] - 0.25) < 1.0e-9 for point in regions[0]["points"]))
            window.register_closed_lines_as_region()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["geometry"]["regions"]), 1)
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_workflow_marks_current_issue_tabs_and_blocks_execution_navigation(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def tab_color_name(window: object, widget: object) -> str:
            tabs = window.tabs
            return tabs.tabBar().tabTextColor(tabs.indexOf(widget)).name().lower()

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            window._load_cfg(
                {
                    "analysis": {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN"},
                    "mesh": {"generator": "rectangle", "nx": 0, "ny": 2, "element_type": "QUAD4", "material": "soil"},
                    "materials": {},
                    "boundary_conditions": [],
                    "loads": [],
                },
                keep_yaml=False,
            )
            window._select_tree_panel("mesh")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "mesh")
            self.assertTrue(window.tree.currentItem().font(0).bold())
            self.assertEqual(window.workflow_ribbon_step_buttons[2].property("workflowStatus"), "current_missing")
            self.assertEqual(window.workflow_guidance_table.currentRow(), 2)
            self.assertEqual(tab_color_name(window, window.form_workspace_stack), "#b42318")
            self.assertEqual(tab_color_name(window, window.check_table), "#b42318")
            self.assertFalse(window.primary_run_button.isEnabled())
            self.assertFalse(window.aux_run_button.isEnabled())
            self.assertIn("入力課題", window.primary_run_button.toolTip())
            presented_mesh_nx = window._present_input_reference("mesh.nx")
            self.assertIn(presented_mesh_nx, window.primary_run_button.toolTip())
            self.assertIn(presented_mesh_nx, window.workflow_ribbon_step_buttons[2].toolTip())
            self.assertIn(presented_mesh_nx, window.aux_panel_stats.text())
            window._select_tree_panel("solver")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "solver")
            self.assertEqual(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole), "solver")
            self.assertTrue(window.tree.currentItem().font(0).bold())
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["solver"])
            self.assertTrue(window.aux_solver_action_group.isVisible())
            self.assertFalse(window.primary_run_button.isEnabled())
            window.workflow_ribbon_step_buttons[8].click()
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "solver")
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_model_check_issue_row_jumps_to_fix_location(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[str] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
            window._activate_panel("model_check")
            for _ in range(200):
                QApplication.processEvents()
                if not getattr(window, "_model_check_job_id", ""):
                    break
            window._apply_model_check_issues(
                [
                    ("ERROR", "materials.soil.gamma", "gamma is invalid", {"path": "materials.soil.gamma"}),
                    ("ERROR", "mesh.quality.area", "bad element", {"elements": ["1"]}),
                ]
            )

            window.model_check_issue_table.selectRow(0)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "materials")
            self.assertTrue(window.inline_detail_sheet_group.isVisible())
            self.assertEqual(window.material_table.currentRow(), 0)
            self.assertEqual(window.material_table.currentColumn(), 4)

            window._activate_panel("model_check")
            window.model_check_issue_table.selectRow(1)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "mesh")
            selected_elements = {
                str(item.data(0).get("id"))
                for item in window.scene.selectedItems()
                if isinstance(item.data(0), dict) and item.data(0).get("kind") == "element"
            }
            self.assertIn("1", selected_elements)
            checks.append("ok")
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, ["ok"])

    def test_workflow_warning_only_materials_and_loads_do_not_stay_red(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QTableWidgetItem
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
            window.check_table.setRowCount(3)
            for row, values in enumerate(
                (
                    ("WARN", "materials.soil.gamma", "単位体積重量を確認してください。"),
                    ("WARN", "loads[1]", "拘束済み境界DOFへの荷重を確認してください。"),
                    ("WARN", "boundary_conditions.rigid_modes[1]", "拘束なしの剛体モード候補です。"),
                )
            ):
                for col, text in enumerate(values):
                    window.check_table.setItem(row, col, QTableWidgetItem(text))
            window._select_tree_panel("materials")
            window.refresh_workflow_guidance()
            QApplication.processEvents()

            self.assertFalse(window.input_execution_blockers())
            self.assertEqual(window.workflow_ribbon_step_buttons[3].property("workflowStatus"), "current")
            self.assertEqual(window.workflow_ribbon_step_buttons[5].property("workflowStatus"), "ok")
            self.assertIn("材料確認: materials.soil.gamma", "\n".join(window.workflow_step_issue_lines("materials", "materials")))
            self.assertIn("WARN: materials.soil.gamma", "\n".join(window.workflow_step_issue_lines("materials", "materials")))
            self.assertIn("荷重確認: loads[1]", "\n".join(window.workflow_step_issue_lines("loads", "loads")))
            self.assertIn("WARN: loads[1]", "\n".join(window.workflow_step_issue_lines("loads", "loads")))
            self.assertIn(
                "拘束なしの剛体モード候補: component 1",
                "\n".join(window.workflow_step_issue_lines("boundary_conditions", "boundary_conditions")),
            )
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_workflow_next_opens_result_summary_before_post_and_report(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QGraphicsLineItem
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "run_20260524_010203"
            results_dir = run_dir / "results"
            stage_dir = results_dir / "Stage-1"
            stage_dir_2 = results_dir / "Stage-2"
            stage_dir.mkdir(parents=True)
            stage_dir_2.mkdir(parents=True)
            summary = results_dir / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "stages": [
                            {"name": "Stage-1", "output_dir": str(stage_dir), "time": 0.0},
                            {"name": "Stage-2", "output_dir": str(stage_dir_2), "time": 10.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (results_dir / "calculation_report.html").write_text("<html><body>report ok</body></html>", encoding="utf-8")
            for current_stage, stress, pressure, sigma_x, sigma_y, tau_xy, ux in (
                (stage_dir, "3.25", "1.75", "11.0", "4.0", "0.35", "0.01"),
                (stage_dir_2, "6.5", "2.5", "22.0", "8.0", "0.7", "0.02"),
            ):
                with (current_stage / "element_stress.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["element_id", "x", "y", "q", "p", "sigma_x", "sigma_y", "tau_xy", "plastic", "active", "material"])
                    writer.writeheader()
                    writer.writerow({
                        "element_id": "1",
                        "x": "0.5",
                        "y": "0.5",
                        "q": stress,
                        "p": pressure,
                        "sigma_x": sigma_x,
                        "sigma_y": sigma_y,
                        "tau_xy": tau_xy,
                        "plastic": "0.0",
                        "active": "1",
                        "material": "soil",
                    })
                with (current_stage / "displacements.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["node_id", "ux", "uy", "u_norm"])
                    writer.writeheader()
                    writer.writerow({"node_id": "1", "ux": ux, "uy": "0.0", "u_norm": ux})

            def fake_exec(app: QApplication) -> int:
                windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
                self.assertTrue(windows)
                window = windows[-1]
                window.project_root = root
                window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
                window.last_run_dir = run_dir
                window._load_result_summary(summary)
                requested: list[str] = []

                def load_sync(kind: str) -> None:
                    requested.append(kind)
                    window.load_result_table(kind)

                window.load_result_table_async = load_sync
                window.current_panel_key = "solver"
                if "solver" in window.panel_pages:
                    window.panel_stack.setCurrentWidget(window.panel_pages["solver"])
                window.refresh_workflow_guidance()
                QApplication.processEvents()
                window.workflow_ribbon_jump_button.click()
                QApplication.processEvents()
                self.assertEqual(window.current_panel_key, "results")
                self.assertEqual(requested, [])
                self.assertIs(window.tabs.currentWidget(), window.panel_stack)
                self.assertEqual(window.results_tabs.currentIndex(), 0)
                self.assertTrue(window.result_judgment_panel.isVisible())
                self.assertEqual(window.result_judgment_model["kind"], "analysis")
                self.assertFalse(window.result_detail_actions_widget.isVisible())
                window.result_more_button.click()
                QApplication.processEvents()
                self.assertTrue(window.result_detail_actions_widget.isVisible())
                self.assertTrue(window.result_table_component_widget.isVisible())
                window.result_more_button.click()
                QApplication.processEvents()
                self.assertFalse(window.result_detail_actions_widget.isVisible())
                window.result_primary_visual_button.click()
                QApplication.processEvents()
                self.assertEqual(requested[:1], ["element_stress"])
                self.assertEqual(window.post_mode, "contour")
                self.assertAlmostEqual(window.result_element_values["1"], 6.5)
                self.assertIn("1", window.result_displacements)
                self.assertEqual(window.tabs.currentIndex(), 0)
                window.aux_deformation_scale_slider.setValue(25)
                QApplication.processEvents()
                self.assertEqual(window.deformation_scale.text(), "25")
                self.assertIn("実変位最大", window.aux_deformation_scale_hint.text())
                self.assertIn("表示変位", window.aux_deformation_scale_hint.text())
                deformed_edges = [
                    item
                    for item in window.scene.items()
                    if isinstance(item, QGraphicsLineItem)
                    and isinstance(item.data(0), dict)
                    and item.data(0).get("kind") == "deformed_edge"
                ]
                deformed_colored_elements = [
                    item
                    for item in window.scene.items()
                    if isinstance(item.data(0), dict)
                    and item.data(0).get("kind") == "deformed_element"
                    and item.data(0).get("component") == "q"
                ]
                self.assertGreater(len(deformed_edges), 0)
                self.assertGreater(len(deformed_colored_elements), 0)
                component_items = [window.aux_result_component_combo.itemText(index) for index in range(window.aux_result_component_combo.count())]
                for component in ("p", "sigma_x", "sigma_y", "tau_xy"):
                    self.assertIn(component, component_items)
                self.assertNotIn("x", component_items)
                for component, expected_value in (("p", 2.5), ("sigma_x", 22.0), ("sigma_y", 8.0), ("tau_xy", 0.7)):
                    window.aux_result_component_combo.setCurrentText(component)
                    QApplication.processEvents()
                    self.assertEqual(window.post_component, component)
                    self.assertEqual(requested[-1], "element_stress")
                    self.assertAlmostEqual(window.result_element_values["1"], expected_value)
                    self.assertIn("1", window.result_displacements)
                    deformed_component_elements = [
                        item
                        for item in window.scene.items()
                        if isinstance(item.data(0), dict)
                        and item.data(0).get("kind") == "deformed_element"
                        and item.data(0).get("component") == component
                    ]
                    self.assertGreater(len(deformed_component_elements), 0)
                window.workflow_ribbon_jump_button.click()
                QApplication.processEvents()
                self.assertEqual(window.current_panel_key, "report")
                self.assertIn("calculation_report.html", window.report_summary.text())
                result_item = window._find_tree_panel_item(window.tree.topLevelItem(0), "result:displacement_vectors")
                self.assertIsNotNone(result_item)
                window.tree.setCurrentItem(result_item)
                QApplication.processEvents()
                self.assertEqual(window.current_panel_key, "results")
                self.assertEqual(window.current_result_kind, "displacement_vectors")
                self.assertEqual(requested[-1], "displacement_vectors")
                self.assertEqual(window.post_mode, "vector")
                self.assertTrue(window.aux_result_control_group.isVisible())
                self.assertFalse(window.aux_action_group.isVisible())
                self.assertEqual(window.aux_result_stage_slider.maximum(), 1)
                self.assertIn("Stage 2", window.aux_result_time_label.text())
                window.aux_result_stage_slider.setValue(0)
                QApplication.processEvents()
                self.assertEqual(window.result_stage_dir.resolve(), stage_dir.resolve())
                self.assertEqual(requested[-1], "displacement_vectors")
                self.assertIn("Stage 1", window.aux_result_time_label.text())
                if window.aux_result_colormap_combo.findText("Terrain") < 0:
                    window.aux_result_colormap_combo.addItem("Terrain")
                window.aux_result_colormap_combo.setCurrentText("Terrain")
                QApplication.processEvents()
                self.assertEqual(window.result_colormap_name, "Terrain")
                checks.append({"requested": requested, "post_mode": window.post_mode})
                window.close()
                return 0

            QApplication.exec = fake_exec
            try:
                self.assertEqual(main_window_module.run_gui(), 0)
            finally:
                QApplication.exec = original_exec

        self.assertEqual(checks[0]["requested"][:1], ["element_stress"])
        self.assertEqual(checks[0]["post_mode"], "vector")

    def test_workflow_report_step_opens_lazy_report_without_existing_html(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "run_20260531_010203"
            results_dir = run_dir / "results"
            stage_dir = results_dir / "Stage-1"
            stage_dir.mkdir(parents=True)
            summary = results_dir / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "stages": [{"name": "Stage-1", "output_dir": str(stage_dir), "time": 0.0}],
                        "output_generation": {"lazy_reports": True},
                        "calculation_report": {"deferred": True},
                        "standard_report": {"deferred": True},
                        "result_view_index": {"deferred": True},
                    }
                ),
                encoding="utf-8",
            )

            def fake_exec(app: QApplication) -> int:
                windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
                self.assertTrue(windows)
                window = windows[-1]
                window.project_root = root
                window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
                window.last_run_dir = run_dir
                window._load_result_summary(summary)
                window.current_panel_key = "results"
                window.refresh_workflow_guidance()
                QApplication.processEvents()

                steps = [step for step in window.workflow_guidance.get("steps", []) if isinstance(step, dict)]
                report_index = next(index for index, step in enumerate(steps) if step.get("panel") == "report")
                report_button = window.workflow_ribbon_step_buttons[report_index]
                self.assertTrue(report_button.isEnabled())
                report_button.click()
                QApplication.processEvents()

                self.assertEqual(window.current_panel_key, "report")
                self.assertIs(window.tabs.currentWidget(), window.panel_stack)
                self.assertIn("Deferred report artifacts", window.report_summary.text())
                self.assertFalse((results_dir / "calculation_report.html").exists())
                self.assertEqual(window._post_export_job_id, "")
                checks.append({"panel": window.current_panel_key})
                window.close()
                return 0

            QApplication.exec = fake_exec
            try:
                self.assertEqual(main_window_module.run_gui(), 0)
            finally:
                QApplication.exec = original_exec

        self.assertEqual(checks, [{"panel": "report"}])

    def test_result_presence_keeps_input_navigation_available_read_only_until_reset(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)
            window.last_run_dir = Path("dummy_run")
            window.current_panel_key = "solver"
            window.panel_stack.setCurrentWidget(window.panel_pages["solver"])
            window._refresh_tree(select_panel="solver")
            window.refresh_workflow_guidance()
            QApplication.processEvents()

            root = window.tree.topLevelItem(0)
            geometry_item = window._find_tree_panel_item(root, "geometry")
            solver_item = window._find_tree_panel_item(root, "solver")
            self.assertIsNotNone(geometry_item)
            self.assertIsNotNone(solver_item)
            self.assertFalse(geometry_item.isDisabled())
            self.assertFalse(solver_item.isDisabled())
            self.assertTrue(window.workflow_ribbon_step_buttons[1].isEnabled())
            self.assertTrue(window.workflow_ribbon_step_buttons[8].isEnabled())
            self.assertTrue(window.workflow_ribbon_reference_label.isVisible())
            self.assertTrue(window.yaml_editor.isReadOnly())

            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertEqual(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole), "geometry")

            window.reset_analysis_results()
            QApplication.processEvents()
            root = window.tree.topLevelItem(0)
            geometry_item = window._find_tree_panel_item(root, "geometry")
            self.assertFalse(geometry_item.isDisabled())
            self.assertFalse(window.workflow_ribbon_reference_label.isVisible())
            self.assertFalse(window.yaml_editor.isReadOnly())
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "geometry")
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_result_reset_and_geometry_coordinate_palette_keep_input_editable(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def select_scene_item(window: object, *, kind: str, ident: str, endpoint: str = "") -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if not isinstance(data, dict):
                    continue
                if data.get("kind") != kind or str(data.get("id")) != ident:
                    continue
                if endpoint and data.get("endpoint") != endpoint:
                    continue
                item.setSelected(True)
                return
            self.fail(f"scene item not found: {kind} {ident} {endpoint}")

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {"lines": [{"id": "L1", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]}]}
            window._load_cfg(cfg, keep_yaml=False)
            window.last_run_dir = Path("dummy_run")
            window.result_stage_dirs = [Path("dummy_run/results/Stage-1")]
            window.result_stage_dir = window.result_stage_dirs[0]
            window.post_mode = "contour"
            window.result_element_values = {"1": 3.0}
            window.result_displacements = {"1": (0.1, 0.0)}
            window.result_rows = [{"element_id": "1", "q": "3.0"}]
            window.result_headers = ["element_id", "q"]
            window.result_table.setRowCount(1)
            window.result_table.setColumnCount(2)
            window.results_summary.setText("dummy result")
            window.reset_analysis_results()
            QApplication.processEvents()
            self.assertIsNone(window.last_run_dir)
            self.assertIsNone(window.result_stage_dir)
            self.assertEqual(window.post_mode, "none")
            self.assertEqual(window.result_element_values, {})
            self.assertEqual(window.result_displacements, {})
            self.assertEqual(window.result_table.rowCount(), 0)
            self.assertEqual(window.cfg["geometry"]["lines"][0]["id"], "L1")
            self.assertIn("入力データは維持", window.results_summary.text())

            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertTrue(hasattr(window, "geometry_point_target"))
            select_scene_item(window, kind="geometry_endpoint", ident="L1", endpoint="start")
            window.load_selected_geometry_point_to_editor()
            self.assertEqual(window.geometry_point_x.text(), "0")
            window.geometry_point_x.setText("0.25")
            window.geometry_point_y.setText("0.5")
            window.apply_geometry_point_absolute_edit()
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["start"], [0.25, 0.5])

            select_scene_item(window, kind="geometry_line", ident="L1")
            window.geometry_point_target.setCurrentIndex(window.geometry_point_target.findData("end"))
            window.geometry_nudge_step.setText("0.25")
            window.nudge_selected_geometry_point(axis="x", sign=1.0)
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["end"], [1.25, 0.0])
            window.close()
            checks.append({"ok": True})
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_workflow_marks_optional_stage_with_input_issue_as_issue_color(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QTableWidgetItem
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["stages"] = [{"name": "Stage-1", "type": "static", "stress_release": 1.2}]
            window._load_cfg(cfg, keep_yaml=False)
            window.check_table.setRowCount(1)
            for col, text in enumerate(("ERROR", "stages[0].stress_release", "stress_release must be between 0 and 1")):
                window.check_table.setItem(0, col, QTableWidgetItem(text))
            window.refresh_workflow_guidance()
            window._select_tree_panel("stages")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "stages")
            self.assertEqual(window.workflow_ribbon_step_buttons[6].property("workflowStatus"), "current_issue")
            self.assertEqual(window.stage_workspace_tabs.tabBar().tabTextColor(0).name().lower(), "#b42318")
            input_assist_tab = window.tabs.indexOf(window.input_assist_table)
            self.assertNotEqual(window.tabs.tabBar().tabTextColor(input_assist_tab).name().lower(), "#b42318")
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_workflow_keeps_stage_info_and_warnings_non_blocking(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QTableWidgetItem
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["stages"] = [{"name": "Stage-1", "type": "static"}]
            window._load_cfg(cfg, keep_yaml=False)
            window.check_table.setRowCount(2)
            for row, values in enumerate(
                (
                    ("INFO", "stages", "1 ステージ"),
                    ("WARN", "stages[1]", "ステージ差分を確認してください。"),
                )
            ):
                for col, text in enumerate(values):
                    window.check_table.setItem(row, col, QTableWidgetItem(text))
            window.refresh_workflow_guidance()
            window._select_tree_panel("stages")
            QApplication.processEvents()

            self.assertFalse(window.input_execution_blockers())
            self.assertEqual(window.workflow_ribbon_step_buttons[6].property("workflowStatus"), "current")
            self.assertNotEqual(window.stage_workspace_tabs.tabBar().tabTextColor(0).name().lower(), "#b42318")
            self.assertIn("WARN: stages[1]", "\n".join(window.workflow_step_issue_lines("stages", "stages")))
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_cad_palette_grid_and_open_endpoint_blocker(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def scene_items_with_kind(window: object, kind: str) -> list[object]:
            items = []
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == kind:
                    items.append(item)
            return items

        def select_scene_item(window: object, *, kind: str, ident: str, endpoint: str = "") -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if not isinstance(data, dict):
                    continue
                if data.get("kind") != kind or str(data.get("id")) != ident:
                    continue
                if endpoint and data.get("endpoint") != endpoint:
                    continue
                item.setSelected(True)
                return
            self.fail(f"scene item not found: {kind} {ident} {endpoint}")

        def cad_button(window: object, label: str) -> QPushButton:
            for button in window.model_cad_palette_widget.findChildren(QPushButton):
                if button.property("cadTool") and button.property("cadToolLabel") == label:
                    return button
            self.fail(f"CAD palette button not found: {label}")

        def scene_point(window: object, x: float, y: float) -> QPointF:
            return QPointF(x * window.preview_scale + window.preview_ox, window.preview_oy - y * window.preview_scale)

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "closure_tolerance": 1.0e-6,
                "lines": [{"id": "L_open", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]}],
            }
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()

            cad_buttons = [
                button
                for button in window.panel_pages["geometry"].findChildren(QPushButton)
                if button.property("cadTool")
            ]
            top_cad_buttons = [
                button
                for button in window.model_cad_palette_widget.findChildren(QPushButton)
                if button.property("cadTool")
            ]
            self.assertTrue(window.model_cad_palette_widget.isVisible())
            self.assertGreaterEqual(len(cad_buttons), 20)
            self.assertGreaterEqual(len(top_cad_buttons), 20)
            self.assertTrue(all(button.text() == "" for button in cad_buttons))
            self.assertTrue(all(button.accessibleName() for button in cad_buttons))
            self.assertTrue(all(button.accessibleName() in button.toolTip() for button in cad_buttons))
            self.assertTrue(all(button.whatsThis() for button in cad_buttons))
            self.assertTrue(all(button.toolTipDuration() >= 8000 for button in cad_buttons))
            self.assertTrue(all(button.property("customCadIcon") for button in top_cad_buttons))
            self.assertIn("直線", {button.property("cadToolLabel") for button in cad_buttons})
            for label in ("端点結合", "閉合線", "補助線化", "分割線化"):
                self.assertIn(label, {button.property("cadToolLabel") for button in top_cad_buttons})
            self.assertLessEqual(max(button.maximumWidth() for button in cad_buttons), 36)
            self.assertLessEqual(max(button.iconSize().width() for button in cad_buttons), 18)
            self.assertTrue(window.aux_geometry_settings_group.isVisible())
            self.assertTrue(window.aux_mesh_settings_group.isHidden())
            self.assertFalse(window.model_cad_edit_widget.isVisible())
            self.assertFalse(window.model_mesh_edit_widget.isVisible())
            right_pane_geometry_buttons = {button.text() for button in window.aux_geometry_settings_group.findChildren(QPushButton)}
            self.assertIn("座標へ移動", right_pane_geometry_buttons)
            self.assertIn("閉合線", right_pane_geometry_buttons)
            cad_button(window, "直線").click()
            QApplication.processEvents()
            self.assertFalse(window.model_cad_mode_label.isVisible())
            self.assertTrue(cad_button(window, "直線").isChecked())
            self.assertEqual(window._current_gui_operation_mode(), "standard")
            self.assertFalse(hasattr(window, "aux_open_detail_button"))
            window.open_current_detail_panel()
            QApplication.processEvents()
            self.assertEqual(window._current_gui_operation_mode(), "standard")
            self.assertEqual(window._detail_return_panel, "geometry")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertTrue(window.model_cad_palette_widget.isVisible())
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)

            self.assertGreater(len(scene_items_with_kind(window, "grid")), 0)
            window.show_grid_lines.setChecked(False)
            QApplication.processEvents()
            self.assertEqual(scene_items_with_kind(window, "grid"), [])
            window.show_grid_lines.setChecked(True)
            QApplication.processEvents()
            self.assertGreater(len(scene_items_with_kind(window, "grid")), 0)

            open_handles = [item for item in scene_items_with_kind(window, "geometry_endpoint") if item.data(0).get("open")]
            self.assertEqual(len(open_handles), 2)
            self.assertTrue(all("未閉合端点" in item.toolTip() for item in open_handles))
            self.assertIn("未閉合端点 2 件", window.geometry_closure_summary.text())
            blockers = window.input_execution_blockers()
            self.assertTrue(any(blocker.get("source") == "geometry_closure" for blocker in blockers))
            self.assertIn(
                window._present_input_reference("geometry.closure"),
                "\n".join(window.workflow_step_issue_lines("geometry", "geometry", blockers)),
            )
            self.assertFalse(window.primary_run_button.isEnabled())

            repair_cfg = plane_strain_quad4_sample()
            repair_cfg["geometry"] = {
                "repair_tolerance": 0.2,
                "closure_tolerance": 1.0e-6,
                "lines": [
                    {"id": "A", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]},
                    {"id": "B", "purpose": "model", "start": [1.05, 0.0], "end": [2.0, 0.0]},
                ],
            }
            window._load_cfg(repair_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            select_scene_item(window, kind="geometry_endpoint", ident="A", endpoint="end")
            window._sync_model_geometry_editor_from_selection()
            self.assertEqual(window.model_geometry_point_x.text(), "1")
            window.model_geometry_point_x.setText("1.02")
            window.model_geometry_point_y.setText("0.0")
            window.apply_geometry_point_absolute_edit(control_scope="model")
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["end"], [1.02, 0.0])
            select_scene_item(window, kind="geometry_endpoint", ident="A", endpoint="end")
            cad_button(window, "端点結合").click()
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["end"], [1.05, 0.0])
            select_scene_item(window, kind="geometry_line", ident="A")
            cad_button(window, "補助線化").click()
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["purpose"], "helper")
            select_scene_item(window, kind="geometry_line", ident="A")
            cad_button(window, "分割線化").click()
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["purpose"], "model")

            closure_cfg = plane_strain_quad4_sample()
            closure_cfg["geometry"] = {
                "repair_tolerance": 0.2,
                "closure_tolerance": 1.0e-6,
                "lines": [
                    {"id": "C", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]},
                    {"id": "D", "purpose": "model", "start": [0.08, 0.0], "end": [1.2, 0.0]},
                ],
            }
            window._load_cfg(closure_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            select_scene_item(window, kind="geometry_endpoint", ident="C", endpoint="start")
            cad_button(window, "閉合線").click()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["geometry"]["lines"]), 3)
            self.assertEqual(window.cfg["geometry"]["lines"][-1]["source"], "closure_helper")

            diagnostic_cfg = plane_strain_quad4_sample()
            diagnostic_cfg["geometry"] = {
                "repair_tolerance": 0.1,
                "closure_tolerance": 1.0e-6,
                "lines": [
                    {"id": "X", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 1.0]},
                    {"id": "Y", "purpose": "model", "start": [0.0, 1.0], "end": [1.0, 0.0]},
                    {"id": "D1", "purpose": "model", "start": [3.0, 0.0], "end": [4.0, 0.0]},
                    {"id": "D2", "purpose": "model", "start": [4.0, 0.0], "end": [3.0, 0.0]},
                    {"id": "S", "purpose": "model", "start": [5.0, 0.0], "end": [5.001, 0.0]},
                    {"id": "G1", "purpose": "model", "start": [6.0, 0.0], "end": [7.0, 0.0]},
                    {"id": "G2", "purpose": "model", "start": [7.05, 0.0], "end": [8.0, 0.0]},
                ],
            }
            window._load_cfg(diagnostic_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            diagnostics = window.refresh_cad_repair_diagnostics(write=True)
            kinds = {row["kind"] for row in diagnostics["candidates"]}
            self.assertTrue({"duplicate_line", "self_intersection", "short_line", "endpoint_gap"}.issubset(kinds))
            self.assertIn("cad_repair_diagnostics", window.cfg["geometry"])
            self.assertGreaterEqual(window.geometry_repair_table.rowCount(), 4)
            self.assertIn("X", window._geometry_repair_line_ids)
            red_lines = []
            for item in scene_items_with_kind(window, "geometry_line"):
                data = item.data(0)
                if isinstance(data, dict) and data.get("id") == "X":
                    red_lines.append(item.pen().color().name().lower())
            self.assertIn("#dc3545", red_lines)
            repair_blockers = window.input_execution_blockers()
            self.assertTrue(any(blocker.get("source") == "cad_repair" for blocker in repair_blockers))
            for row in range(window.geometry_repair_table.rowCount()):
                if window._table_text(window.geometry_repair_table, row, 1) == "duplicate_line":
                    window.geometry_repair_table.selectRow(row)
                    window.select_cad_repair_candidate()
                    QApplication.processEvents()
                    selected_geometry = [
                        item.data(0).get("id")
                        for item in window.scene.selectedItems()
                        if isinstance(item.data(0), dict) and item.data(0).get("kind") == "geometry_line"
                    ]
                    self.assertTrue({"D1", "D2"}.intersection(set(selected_geometry)))
                    break
            else:
                self.fail("duplicate_line repair candidate row not found")

            window._select_tree_panel("mesh")
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertIn("入力課題", window.statusBar().currentMessage())

            helper_cfg = plane_strain_quad4_sample()
            helper_cfg["geometry"] = {
                "lines": [{"id": "L_helper", "purpose": "helper", "start": [0.0, 0.0], "end": [1.0, 0.0]}]
            }
            window._load_cfg(helper_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertFalse(any(blocker.get("source") == "geometry_closure" for blocker in window.input_execution_blockers()))
            self.assertEqual([item for item in scene_items_with_kind(window, "geometry_endpoint") if item.data(0).get("open")], [])
            self.assertIn("閉合しています", window.geometry_closure_summary.text())

            drawing_cfg = plane_strain_quad4_sample()
            drawing_cfg["geometry"] = {"lines": []}
            window._load_cfg(drawing_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            window.set_draw_mode("polyline")
            window.handle_model_click(scene_point(window, 0.0, 0.0))
            window.handle_model_click(scene_point(window, 1.0, 0.0))
            window.handle_model_click(scene_point(window, 1.0, 1.0))
            window.finish_cad_drawing()
            self.assertGreaterEqual(sum(1 for line in window.cfg["geometry"]["lines"] if line.get("source") == "cad_polyline"), 2)
            window.set_draw_mode("rectangle")
            window.handle_model_click(scene_point(window, 2.0, 0.0))
            window.handle_model_click(scene_point(window, 3.0, 1.0))
            self.assertTrue(any(region.get("source") == "cad_rectangle" for region in window.cfg["geometry"]["regions"]))
            window.set_draw_mode("circle")
            window.handle_model_click(scene_point(window, 4.0, 0.0))
            window.handle_model_click(scene_point(window, 5.0, 0.0))
            self.assertTrue(any(tunnel.get("source") == "cad_circle" for tunnel in window.cfg["geometry"]["tunnels"]))
            window.set_draw_mode("arc")
            for x, y in ((6.0, 0.0), (6.5, 0.5), (7.0, 0.0)):
                window.handle_model_click(scene_point(window, x, y))
            self.assertTrue(any(arc.get("source") == "cad_arc" for arc in window.cfg["geometry"]["arcs"]))
            arc_id = window.cfg["geometry"]["arcs"][-1]["id"]
            self.assertTrue(
                window._set_cad_draw_control_point_from_drag(
                    {"kind": "cad_draw_control_point", "entity": "arc", "id": arc_id, "point": 2},
                    6.5,
                    0.7,
                )
            )
            self.assertTrue(any(line.get("source_ref") == arc_id for line in window.cfg["geometry"]["lines"]))
            window.set_draw_mode("curve")
            for x, y in ((8.0, 0.0), (8.4, 0.3), (9.0, 0.0)):
                window.handle_model_click(scene_point(window, x, y))
            window.finish_cad_drawing()
            self.assertTrue(any(curve.get("source") == "cad_curve" for curve in window.cfg["geometry"]["curves"]))
            curve_id = window.cfg["geometry"]["curves"][-1]["id"]
            self.assertTrue(
                window._set_cad_draw_control_point_from_drag(
                    {"kind": "cad_draw_control_point", "entity": "curve", "id": curve_id, "point": 2},
                    8.5,
                    0.6,
                )
            )
            self.assertEqual(window.cfg["geometry"]["curves"][-1]["control_points"][1], [8.5, 0.6])
            window.set_draw_mode("point")
            window.handle_model_click(scene_point(window, 9.5, 0.25))
            self.assertTrue(window.cfg["geometry"]["points"])
            QApplication.processEvents()
            self.assertTrue(scene_items_with_kind(window, "geometry_point"))
            window.set_draw_mode("line")
            window.handle_model_click(scene_point(window, 10.0, 0.0))
            window.model_cad_length_edit.setText("2")
            window.model_cad_angle_edit.setText("0")
            window.handle_model_click(scene_point(window, 10.0, 5.0))
            self.assertEqual(window.cfg["geometry"]["lines"][-1]["end"], [12.0, 0.0])
            window.model_cad_length_edit.clear()
            window.model_cad_angle_edit.clear()
            endpoint_cfg = plane_strain_quad4_sample()
            endpoint_cfg["geometry"] = {
                "lines": [{"id": "existing", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]}]
            }
            window._load_cfg(endpoint_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            window.set_draw_mode("line")
            endpoint_item = next(item for item in scene_items_with_kind(window, "geometry_endpoint") if item.data(0).get("id") == "existing" and item.data(0).get("endpoint") == "end")
            endpoint_pos = window.view.mapFromScene(endpoint_item.sceneBoundingRect().center())
            window.view.mousePressEvent(type("FakeMouseEvent", (), {
                "button": lambda self: Qt.MouseButton.LeftButton,
                "pos": lambda self, p=endpoint_pos: p,
                "accept": lambda self: None,
            })())
            self.assertEqual(window.draw_start, (1.0, 0.0))
            window.set_draw_mode("line")
            window.handle_model_click(scene_point(window, 13.0, 0.0))
            window.update_cad_lightweight_preview(scene_point(window, 14.0, 0.0))
            self.assertTrue(scene_items_with_kind(window, "cad_preview"))
            window.finish_cad_drawing()
            self.assertFalse(scene_items_with_kind(window, "cad_preview"))
            window.model_cad_command_edit.setText("line 15,0,16,0")
            window.apply_cad_command_input()
            self.assertEqual(window.cfg["geometry"]["lines"][-1]["start"], [15.0, 0.0])
            vertex_line_id = str(window.cfg["geometry"]["lines"][-1]["id"])
            select_scene_item(window, kind="geometry_line", ident=vertex_line_id)
            before_vertex_lines = len(window.cfg["geometry"]["lines"])
            window.model_cad_command_edit.setText("addvertex 15.5,0")
            window.apply_cad_command_input()
            self.assertEqual(len(window.cfg["geometry"]["lines"]), before_vertex_lines + 1)
            split_line_id = str(window.cfg["geometry"]["lines"][-1]["id"])
            QApplication.processEvents()
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "geometry_line" and str(data.get("id")) in {vertex_line_id, split_line_id}:
                    item.setSelected(True)
            window.model_cad_command_edit.setText("delvertex")
            window.apply_cad_command_input()
            self.assertFalse(any(str(line.get("id")) == split_line_id for line in window.cfg["geometry"]["lines"]))
            window.cfg["geometry"].setdefault("points", []).extend(
                [
                    {"id": "snap_a", "point": [20.0, 0.0]},
                    {"id": "snap_b", "point": [20.03, 0.0]},
                ]
            )
            window.snap_type.setCurrentIndex(window.snap_type.findData("geometry"))
            window.set_draw_mode("line")
            window.update_cad_lightweight_preview(scene_point(window, 20.015, 0.0))
            self.assertGreaterEqual(len(scene_items_with_kind(window, "cad_snap_candidate")), 2)
            active_before = [
                item.data(0).get("index")
                for item in scene_items_with_kind(window, "cad_snap_candidate")
                if item.data(0).get("active")
            ]
            window.cycle_cad_snap_candidate()
            active_after = [
                item.data(0).get("index")
                for item in scene_items_with_kind(window, "cad_snap_candidate")
                if item.data(0).get("active")
            ]
            self.assertNotEqual(active_before, active_after)
            window.snap_type.setCurrentIndex(window.snap_type.findData("all"))
            window.cancel_cad_drawing()

            outline_cfg = plane_strain_quad4_sample()
            window._load_cfg(outline_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            self.assertEqual(window.geometry_line_table.rowCount(), 4)
            self.assertTrue(all(line.get("source") == "mesh_rectangle_outline" for line in window.cfg["geometry"]["lines"]))
            window.open_current_detail_panel()
            QApplication.processEvents()
            self.assertEqual(window._current_gui_operation_mode(), "standard")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertEqual(window.geometry_line_table.rowCount(), 4)
            cad_button(window, "直線").click()
            QApplication.processEvents()
            for point in (scene_point(window, 0.5, 0.5), scene_point(window, 1.5, 0.5)):
                view_pos = window.view.mapFromScene(point)
                window.view.mousePressEvent(
                    type(
                        "FakeMouseEvent",
                        (),
                        {
                            "button": lambda self: Qt.MouseButton.LeftButton,
                            "pos": lambda self, p=view_pos: p,
                            "accept": lambda self: None,
                        },
                    )()
                )
            QApplication.processEvents()
            self.assertEqual(window.geometry_line_table.rowCount(), 5)
            start = window.cfg["geometry"]["lines"][-1]["start"]
            end = window.cfg["geometry"]["lines"][-1]["end"]
            self.assertAlmostEqual(start[0], 0.5, delta=0.01)
            self.assertAlmostEqual(start[1], 0.5, delta=0.01)
            self.assertAlmostEqual(end[0], 1.5, delta=0.01)
            self.assertAlmostEqual(end[1], 0.5, delta=0.01)
            drawn_line_id = str(window.cfg["geometry"]["lines"][-1]["id"])
            cad_button(window, "選択").click()
            QApplication.processEvents()
            self.assertEqual(window.draw_mode, "select")
            self.assertEqual(window.view.dragMode(), type(window.view).DragMode.NoDrag)
            self.assertTrue(cad_button(window, "選択").isChecked())
            click_pos = window.view.mapFromScene(scene_point(window, 1.0, 0.5))
            window.view.mousePressEvent(
                type(
                    "FakeMouseEvent",
                    (),
                    {
                        "button": lambda self: Qt.MouseButton.LeftButton,
                        "pos": lambda self, p=click_pos: p,
                        "modifiers": lambda self: Qt.KeyboardModifier.NoModifier,
                        "accept": lambda self: None,
                    },
                )()
            )
            QApplication.processEvents()
            selected_line_ids = {
                str(item.data(0).get("id"))
                for item in window.scene.selectedItems()
                if isinstance(item.data(0), dict) and item.data(0).get("kind") == "geometry_line"
            }
            self.assertIn(drawn_line_id, selected_line_ids)
            selected_endpoint_ids = {
                str(item.data(0).get("id"))
                for item in window.scene.selectedItems()
                if isinstance(item.data(0), dict) and item.data(0).get("kind") == "geometry_endpoint"
            }
            self.assertIn(drawn_line_id, selected_endpoint_ids)
            window.delete_selected_geometry()
            QApplication.processEvents()
            self.assertFalse(any(str(line.get("id")) == drawn_line_id for line in window.cfg["geometry"]["lines"]))
            self.assertEqual(window.geometry_line_table.rowCount(), 4)
            endpoint_line_id = str(window.cfg["geometry"]["lines"][0]["id"])
            select_scene_item(window, kind="geometry_endpoint", ident=endpoint_line_id, endpoint="start")
            window.delete_selected_geometry()
            QApplication.processEvents()
            self.assertFalse(any(str(line.get("id")) == endpoint_line_id for line in window.cfg["geometry"]["lines"]))
            self.assertEqual(window.geometry_line_table.rowCount(), 3)
            table_line_id = str(window.cfg["geometry"]["lines"][0]["id"])
            for row in range(window.geometry_line_table.rowCount()):
                if window._table_text(window.geometry_line_table, row, 0) == table_line_id:
                    window.geometry_line_table.selectRow(row)
                    break
            else:
                self.fail("line row not found for detail deletion")
            window.remove_selected_geometry_rows(window.geometry_line_table)
            QApplication.processEvents()
            self.assertFalse(any(str(line.get("id")) == table_line_id for line in window.cfg["geometry"]["lines"]))
            self.assertEqual(window.geometry_line_table.rowCount(), 2)
            self.assertFalse(
                any(
                    isinstance(item.data(0), dict)
                    and item.data(0).get("kind") == "geometry_line"
                    and str(item.data(0).get("id")) == table_line_id
                    for item in window.scene.items()
                )
            )
            sync_cfg = plane_strain_quad4_sample()
            sync_cfg["geometry"] = {"lines": []}
            window._load_cfg(sync_cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()
            window.model_cad_command_edit.setText("polyline 0,0,1,0,2,0")
            window.apply_cad_command_input()
            self.assertEqual(window.geometry_line_table.rowCount(), 2)
            self.assertTrue(all(line.get("source") == "cad_command_polyline" for line in window.cfg["geometry"]["lines"]))
            window.open_current_detail_panel()
            QApplication.processEvents()
            self.assertEqual(window._current_gui_operation_mode(), "standard")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertEqual(window.geometry_line_table.rowCount(), 2)
            self.assertEqual(window._table_text(window.geometry_line_table, 0, 5), "1.0")
            self.assertFalse(window._geometry_detail_tables_dirty)
            self.assertTrue(all(line.get("source") == "cad_command_polyline" for line in window.cfg["geometry"]["lines"]))
            window.geometry_line_table.item(0, 5).setText("1.5")
            QApplication.processEvents()
            window._geometry_detail_tables_dirty = True
            self.assertTrue(window._geometry_detail_tables_dirty)
            self.assertTrue(window._apply_geometry_detail_to_cfg_if_dirty())
            QApplication.processEvents()
            self.assertEqual(window.cfg["geometry"]["lines"][0]["end"], [1.5, 0.0])
            self.assertEqual(window.cfg["geometry"]["lines"][0].get("source"), "cad_command_polyline")
            self.assertTrue(window.model_cad_palette_widget.isVisible())

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_drawing_palette_chip_tools_cover_toolbar_actions_and_disable_unusable_actions(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def chip(container: object, label: str) -> QPushButton:
            for button in container.findChildren(QPushButton):
                if button.property("cadTool") and button.property("cadToolLabel") == label:
                    return button
            self.fail(f"palette chip not found: {label}")

        def select_scene_item(window: object, *, kind: str, ident: str, endpoint: str = "") -> None:
            window.scene.clearSelection()
            for item in window.scene.items():
                data = item.data(0)
                if not isinstance(data, dict):
                    continue
                if data.get("kind") != kind or str(data.get("id")) != ident:
                    continue
                if endpoint and data.get("endpoint") != endpoint:
                    continue
                item.setSelected(True)
                QApplication.processEvents()
                return
            self.fail(f"scene item not found: {kind} {ident} {endpoint}")

        def assert_disabled(button: QPushButton, reason_part: str = "無効") -> None:
            self.assertFalse(button.isEnabled(), button.accessibleName())
            self.assertTrue(button.property("chipDisabled"))
            self.assertIn("#f1f5f9", button.styleSheet())
            self.assertIn(reason_part, button.toolTip())

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {
                "lines": [{"id": "L1", "purpose": "model", "start": [0.0, 0.0], "end": [1.0, 0.0]}],
                "regions": [{"id": "R1", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}],
            }
            cfg["mesh"] = {"mode": "auto_mixed", "target_size": 1.0, "division_width": 1.0, "element_type": "QUAD4", "nx": 1, "ny": 1}
            window._load_cfg(cfg, keep_yaml=False)
            window._select_tree_panel("geometry")
            QApplication.processEvents()

            cad_labels = {
                button.property("cadToolLabel")
                for button in window.model_cad_palette_widget.findChildren(QPushButton)
                if button.property("cadTool")
            }
            toolbar_backed_labels = {
                "領域を閉じる",
                "モデルチェック",
                "矩形選択",
                "投げ縄",
                "条件選択",
                "選択解除",
                "反転",
                "履歴記録",
                "名前保存",
                "選択比較",
                "モードヘルプ",
                "選択set登録",
            }
            self.assertTrue(toolbar_backed_labels.issubset(cad_labels))
            self.assertTrue(all(button.property("customCadIcon") for button in window.model_cad_palette_widget.findChildren(QPushButton) if button.property("cadTool")))

            window.scene.clearSelection()
            window._refresh_cad_tool_button_state()
            QApplication.processEvents()
            for label in ("消しゴム", "トリム", "端点結合", "格子条件", "領域を閉じる", "選択解除", "名前保存", "選択set登録", "Undo", "Redo"):
                assert_disabled(chip(window.model_cad_palette_widget, label))

            chip(window.model_cad_palette_widget, "直線").click()
            QApplication.processEvents()
            self.assertEqual(window.draw_mode, "line")
            self.assertTrue(chip(window.model_cad_palette_widget, "直線").isChecked())
            chip(window.model_cad_palette_widget, "矩形選択").click()
            QApplication.processEvents()
            self.assertEqual(window.draw_mode, "select")
            self.assertEqual(window.selection_mode, "rectangle")
            chip(window.model_cad_palette_widget, "投げ縄").click()
            QApplication.processEvents()
            self.assertEqual(window.selection_mode, "lasso")
            chip(window.model_cad_palette_widget, "多角形").click()
            QApplication.processEvents()
            self.assertEqual(window.draw_mode, "region")
            window.region_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
            window._refresh_cad_tool_button_state()
            self.assertTrue(chip(window.model_cad_palette_widget, "領域を閉じる").isEnabled())

            select_scene_item(window, kind="geometry_line", ident="L1")
            self.assertTrue(chip(window.model_cad_palette_widget, "消しゴム").isEnabled())
            self.assertTrue(chip(window.model_cad_palette_widget, "トリム").isEnabled())
            self.assertTrue(chip(window.model_cad_palette_widget, "補助線化").isEnabled())
            assert_disabled(chip(window.model_cad_palette_widget, "端点結合"))
            select_scene_item(window, kind="geometry_endpoint", ident="L1", endpoint="start")
            self.assertTrue(chip(window.model_cad_palette_widget, "端点結合").isEnabled())
            select_scene_item(window, kind="geometry_region", ident="R1")
            self.assertTrue(chip(window.model_cad_palette_widget, "格子条件").isEnabled())

            window._select_tree_panel("mesh")
            QApplication.processEvents()
            mesh_labels = {
                button.property("cadToolLabel")
                for button in window.model_mesh_palette_widget.findChildren(QPushButton)
                if button.property("cadTool")
            }
            self.assertTrue(toolbar_backed_labels.difference({"領域を閉じる"}).issubset(mesh_labels))
            window.scene.clearSelection()
            window._refresh_mesh_tool_button_state()
            for label in ("削除", "ブロック分割", "違反選択", "違反修復", "選択解除", "名前保存", "選択set登録", "Undo", "Redo"):
                assert_disabled(chip(window.model_mesh_palette_widget, label))
            chip(window.model_mesh_palette_widget, "選択").click()
            QApplication.processEvents()
            self.assertEqual(window.draw_mode, "select")
            self.assertTrue(chip(window.model_mesh_palette_widget, "選択").isChecked())

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, [{"ok": True}])

    def test_job_controller_tracks_cancel_and_completion(self) -> None:
        controller = GuiJobController()
        job_id = controller.start_job("solver", target="input.yaml", metadata={"run_dir": "runs/run_1"})
        self.assertEqual(len(controller.active_jobs()), 1)

        self.assertTrue(controller.request_cancel(job_id))
        self.assertTrue(controller.is_cancel_requested(job_id))
        self.assertTrue(controller.cancel_job(job_id, message="stopped by user"))

        snapshot = controller.snapshot()
        self.assertEqual(snapshot[0]["status"], "cancelled")
        self.assertEqual(snapshot[0]["message"], "stopped by user")
        self.assertEqual(len(controller.active_jobs()), 0)

        fail_id = controller.start_job("autosave", target="autosave")
        self.assertTrue(controller.complete_job(fail_id, status="failed", message="disk full"))
        manifest = controller.failure_manifest(fail_id, message="disk full", context={"button": "autosave"})
        self.assertFalse(manifest["ok"])
        self.assertEqual(manifest["job"]["status"], "failed")
        self.assertEqual(manifest["context"]["button"], "autosave")
        with self.assertRaises(ValueError):
            controller.complete_job(fail_id, status="unknown")

    def test_background_task_normalizes_success_failure_and_cancel(self) -> None:
        ok = run_callable_with_token("job-ok", lambda token: "done")
        self.assertTrue(ok.ok)
        self.assertEqual(ok.value, "done")

        failed = run_callable_with_token("job-fail", lambda token: (_ for _ in ()).throw(ValueError("bad input")))
        self.assertFalse(failed.ok)
        self.assertIn("bad input", failed.error)

        token = CancellationToken()
        token.cancel()
        cancelled = run_callable_with_token("job-cancel", lambda active: "never", token)
        self.assertFalse(cancelled.ok)
        self.assertIn("cancelled", cancelled.error)

    def test_qt_callable_runner_emits_finished_and_failed(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")
        from geofem_app.gui.qt_workers import QtCallableRunner

        _app = QApplication.instance() or QApplication([])
        finished: list[tuple[str, object]] = []
        failed: list[tuple[str, str]] = []

        ok_runner = QtCallableRunner("job-ok", lambda: {"ok": True})
        ok_runner.signals.finished.connect(lambda job_id, value: finished.append((job_id, value)))
        ok_runner.run()

        fail_runner = QtCallableRunner("job-fail", lambda: (_ for _ in ()).throw(ValueError("bad input")))
        fail_runner.signals.failed.connect(lambda job_id, message: failed.append((job_id, message)))
        fail_runner.run()

        self.assertEqual(finished, [("job-ok", {"ok": True})])
        self.assertEqual(failed[0][0], "job-fail")
        self.assertIn("bad input", failed[0][1])

    def test_gui_numba_warmup_runs_as_background_job(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")
        import geofem_app.gui.main_window as main_window_module

        class ImmediatePool:
            def start(self, runner: object) -> None:
                runner.run()

        original_exec = QApplication.exec
        original_warmup = main_window_module.warmup_numba_kernels
        original_enabled = main_window_module.gui_numba_warmup_enabled
        captured: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.gui_thread_pool = ImmediatePool()
            window.start_numba_warmup_async()
            captured.append(dict(window.numba_warmup_summary))
            self.assertEqual(window._numba_warmup_job_id, "")
            window.close()
            return 0

        QApplication.exec = fake_exec
        main_window_module.gui_numba_warmup_enabled = lambda: True
        main_window_module.warmup_numba_kernels = lambda profile="gui": {
            "schema": "geofem.numba_warmup.v1",
            "enabled": True,
            "profile": profile,
            "elapsed_seconds": 0.01,
            "kernel_count": 1,
            "warmed_count": 1,
            "failed_count": 0,
            "kernels": [],
        }
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
            main_window_module.warmup_numba_kernels = original_warmup
            main_window_module.gui_numba_warmup_enabled = original_enabled

        self.assertEqual(captured[0]["schema"], "geofem.numba_warmup.v1")
        self.assertEqual(captured[0]["profile"], "gui")

    def test_autosave_io_writes_latest_and_stamped_files_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = write_autosave_files(
                Path(tmp),
                "recent project",
                "analysis:\n  dimension: 2D\n",
                timestamp=datetime(2026, 5, 23, 8, 0, 0),
            )

            self.assertTrue(result.latest.exists())
            self.assertTrue(result.stamped.exists())
            self.assertEqual(result.latest.read_text(encoding="utf-8"), "analysis:\n  dimension: 2D\n")
            self.assertEqual(result.byte_count, len("analysis:\n  dimension: 2D\n".encode("utf-8")))
            self.assertEqual(result.sha256, hashlib.sha256("analysis:\n  dimension: 2D\n".encode("utf-8")).hexdigest())
            self.assertFalse(list((Path(tmp) / "autosave").glob("*.tmp")))

    def test_watchdog_record_append_keeps_compact_ring_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gui_freeze_watchdog.json"
            for index in range(3):
                record = GuiFreezeWatchdogRecord(
                    timestamp=f"2026-05-23T08:00:0{index}Z",
                    operation="モデルチェック",
                    job="model_check",
                    elapsed_ms=3500.0 + index,
                    delay_ms=2500.0 + index,
                    target_file="model.yaml",
                    line_count=42,
                    mesh_nodes=100,
                    mesh_elements=80,
                    last_ui_event="MouseButtonPress:QPushButton",
                    active_jobs=[{"id": f"job-{index}", "kind": "model_check"}],
                )
                rows = append_watchdog_record(path, record.as_dict(), max_records=2)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["timestamp"], "2026-05-23T08:00:01Z")
            self.assertEqual(rows[1]["active_jobs"][0]["id"], "job-2")
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_display_quality_policy_reduces_large_model_detail(self) -> None:
        small = resolve_display_quality_policy(
            mode="auto",
            node_count=12,
            element_count=8,
            detail_limit=100,
            vector_limit=50,
            requested_node_labels=True,
            requested_element_labels=True,
            requested_element_boundaries=True,
            requested_contour_labels=True,
            requested_contour_levels=12,
            requested_curve_segments=6,
        )
        self.assertFalse(small.reduced)
        self.assertFalse(small.batch_mesh_items)
        self.assertFalse(small.batch_node_items)
        self.assertFalse(small.pick_nearest_mesh)
        self.assertTrue(small.draw_node_labels)
        self.assertEqual(small.contour_level_count, 12)

        large = resolve_display_quality_policy(
            mode="auto",
            node_count=5000,
            element_count=4800,
            detail_limit=1000,
            vector_limit=300,
            requested_node_labels=True,
            requested_element_labels=True,
            requested_element_boundaries=True,
            requested_contour_labels=True,
            requested_contour_levels=20,
            requested_curve_segments=8,
        )
        self.assertTrue(large.reduced)
        self.assertTrue(large.batch_mesh_items)
        self.assertTrue(large.batch_node_items)
        self.assertTrue(large.pick_nearest_mesh)
        self.assertFalse(large.draw_node_labels)
        self.assertFalse(large.draw_element_labels)
        self.assertFalse(large.draw_contour_labels)
        self.assertLessEqual(large.max_vectors, 300)
        self.assertLessEqual(large.contour_level_count, 8)

    def test_mesh_quality_worker_helpers_return_detached_results(self) -> None:
        cfg = plane_strain_quad4_sample()
        violations = collect_mesh_quality_violations_snapshot(cfg)
        candidates = compare_mesh_quality_improvements_snapshot(
            cfg,
            methods=["laplace"],
            iterations=1,
            thresholds=(1.0e-12, 10.0, 20.0, 0.85),
            selected_elements=[],
        )

        self.assertIsInstance(violations, list)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["report"]["method"], "laplace")
        self.assertIn("nodes", candidates[0])

    def test_auto_geometry_mesh_worker_uses_detached_context(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")
        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        results: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            cfg = plane_strain_quad4_sample()
            cfg["geometry"] = {"regions": [{"id": "R1", "points": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]}]}
            cfg["mesh"] = {"mode": "auto_mixed", "target_size": 0.5, "material": "soil"}
            results.append(
                generate_auto_geometry_mesh_snapshot(
                    type(window),
                    cfg,
                    requested_type="QUAD4",
                    material="soil",
                    integration="full",
                )
            )
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        mesh = results[0]["mesh"]
        self.assertIsInstance(mesh, dict)
        self.assertIn("auto_mesh_confirmed_at", mesh)
        self.assertGreater(len(mesh["nodes"]), 0)
        self.assertGreater(len(mesh["elements"]), 0)

    def test_cad_worker_splits_lines_at_intersections(self) -> None:
        result = split_lines_at_intersections_snapshot(
            {
                "lines": [
                    {"id": "a", "start": [0.0, 0.0], "end": [2.0, 0.0]},
                    {"id": "b", "start": [1.0, -1.0], "end": [1.0, 1.0]},
                ]
            }
        )

        self.assertEqual(result["split_count"], 2)
        self.assertEqual(len(result["geometry"]["lines"]), 4)

    def test_cad_dimension_constraint_worker_uses_detached_context(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")
        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        results: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            geometry = {
                "lines": [{"id": "L1", "start": [0.0, 0.0], "end": [2.0, 0.0]}],
                "dimensions": [{"id": "D1", "start": [0.0, 0.0], "end": [2.0, 0.0], "text": "3.0", "constraint": "length"}],
            }
            constraints = [
                {
                    "id": "D1",
                    "type": "length",
                    "locked": False,
                    "start": [0.0, 0.0],
                    "end": [2.0, 0.0],
                    "value": 3.0,
                    "measured": 2.0,
                }
            ]
            results.append(solve_dimension_constraints_snapshot(type(window), geometry, constraints))
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        geometry_out = results[0]["geometry"]
        self.assertIsInstance(geometry_out, dict)
        line = geometry_out["lines"][0]
        self.assertAlmostEqual(line["end"][0], 3.0)
        self.assertEqual(results[0]["constraint_count"], 1)

    def test_post_workers_prepare_display_state_off_gui_thread(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")
        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        results: list[tuple[dict[str, object], dict[str, object]]] = []

        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage_1"
            stage.mkdir()
            stress = stage / "element_stress.csv"
            with stress.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["element_id", "x", "y", "q", "p", "plastic", "FL", "active", "material"])
                writer.writeheader()
                writer.writerow({"element_id": "1", "x": 0.5, "y": 0.5, "q": 1.0, "p": 1.0, "plastic": 1.0, "FL": 0.85, "active": 1, "material": "soil"})

            def fake_exec(app: QApplication) -> int:
                windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
                self.assertTrue(windows)
                window = windows[0]
                cfg = plane_strain_quad4_sample()
                post = load_post_table_snapshot(
                    type(window),
                    cfg,
                    path=stress,
                    kind="safety_factor",
                    result_component="FL",
                    table_component="FL",
                    result_stage_dir=str(stage),
                )
                srm = load_srm_post_snapshot(
                    type(window),
                    cfg,
                    result_stage_dir=str(stage),
                    options={"fl_limit": "1.05", "plastic_threshold": "0.5", "local_fl_aggregation": "mean"},
                )
                results.append((post, srm))
                window.close()
                return 0

            QApplication.exec = fake_exec
            try:
                self.assertEqual(run_gui(), 0)
            finally:
                QApplication.exec = original_exec

        post, srm = results[0]
        self.assertEqual(post["post_mode"], "contour")
        self.assertEqual(post["post_component"], "FL")
        self.assertAlmostEqual(post["result_element_values"]["1"], 0.85)
        self.assertEqual(len(srm["rows"]), 1)
        self.assertAlmostEqual(srm["result_element_values"]["1"], 0.85)
        self.assertIn("SRM安全率", srm["summary_text"])

    def test_post_worker_streams_table_page_and_keeps_contour_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage_1"
            stage.mkdir()
            stress = stage / "element_stress.csv"
            with stress.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["element_id", "x", "y", "q", "p", "plastic", "active"])
                writer.writeheader()
                for index in range(25):
                    writer.writerow(
                        {
                            "element_id": str(index + 1),
                            "x": index,
                            "y": index * 0.5,
                            "q": index * 2.0,
                            "p": index * 0.25,
                            "plastic": 1 if index % 3 == 0 else 0,
                            "active": 1,
                        }
                    )

            post = load_post_table_snapshot(
                object,
                {},
                path=stress,
                kind="element_stress",
                result_component="q",
                table_component="q",
                result_stage_dir=str(stage),
                page_size=7,
            )

        self.assertEqual(len(post["rows"]), 7)
        self.assertEqual(post["table_summary"]["row_count"], 25)
        self.assertEqual(post["table_page"]["page_count"], 4)
        self.assertEqual(post["table_summary"]["minimums"]["q"], 0.0)
        self.assertEqual(post["table_summary"]["maximums"]["q"], 48.0)
        self.assertEqual(len(post["result_element_values"]), 25)
        self.assertAlmostEqual(post["result_element_values"]["25"], 48.0)
        cached_p = materialize_post_component_snapshot(post["component_store"], "p")
        self.assertEqual(cached_p["source"], "compact_component_store")
        self.assertEqual(cached_p["value_count"], 25)
        self.assertAlmostEqual(cached_p["result_element_values"]["25"], 6.0)

    def test_large_post_component_switch_uses_compact_values_without_full_row_retention(self) -> None:
        row_count = 50000
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage_1"
            stage.mkdir()
            stress = stage / "element_stress.csv"
            with stress.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["element_id", "x", "y", "q", "p", "plastic", "active"])
                writer.writeheader()
                for index in range(row_count):
                    writer.writerow(
                        {
                            "element_id": str(index + 1),
                            "x": index,
                            "y": index * 0.5,
                            "q": index * 2.0,
                            "p": index * 0.25,
                            "plastic": index % 2,
                            "active": 1,
                        }
                    )
            post = load_post_table_snapshot(
                object,
                {},
                path=stress,
                kind="element_stress",
                result_component="q",
                table_component="q",
                result_stage_dir=str(stage),
                page_size=128,
            )
            cached_p = materialize_post_component_snapshot(post["component_store"], "p")

        self.assertEqual(len(post["rows"]), 128)
        self.assertEqual(post["component_store"]["row_count"], row_count)
        self.assertLessEqual(post["component_store"]["storage_bytes"], row_count * 5 * 8)
        self.assertEqual(cached_p["value_count"], row_count)
        self.assertAlmostEqual(cached_p["result_element_values"][str(row_count)], (row_count - 1) * 0.25)

    def test_post_export_workers_write_images_pdf_reports_and_audit(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtGui import QImage
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")
        from geofem_app.post_image_diff import create_sample_post_image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = create_sample_post_image(root / "source.png", width=180, height=120)
            image = QImage(str(source))
            saved = save_post_image_snapshot(image.copy(), root / "saved.png", role="baseline")
            self.assertEqual(saved["role"], "baseline")
            self.assertTrue(Path(str(saved["path"])).exists())

            diff = compare_post_image_snapshot(QImage(str(saved["path"])), saved["path"])
            self.assertTrue(diff["ok"])

            pdf = export_scene_pdf_snapshot(root / "post_view.pdf", current_image=image.copy(), layout_specs=[], snapshot_paths=[])
            pdf_path = Path(str(pdf["path"]))
            self.assertTrue(pdf_path.exists())
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-"))

            build_results = root / "build_results"
            stage = root / "stage_1"
            build_results.mkdir()
            stage.mkdir()
            (build_results / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            (build_results / "run.log").write_text("run ok", encoding="utf-8")
            (stage / "displacements.csv").write_text("node_id,ux,uy\n1,0,0\n", encoding="utf-8")
            built = build_selected_report_snapshot(results_dir=build_results, stage_dir=stage, include_summary=True, include_tables=True)
            built_path = Path(str(built["path"]))
            self.assertTrue(built_path.exists())
            self.assertIn("Result table links", built_path.read_text(encoding="utf-8"))

            copied = copy_report_pdf_snapshot(pdf_path, root / "copied_report.pdf", manifest=root / "manifest.json")
            self.assertTrue(Path(str(copied["path"])).exists())

            result_dir = root / "results"
            result_dir.mkdir()
            (result_dir / "calculation_report.html").write_text("<html>report</html>", encoding="utf-8")
            (result_dir / "calculation_report.pdf").write_bytes(b"%PDF-1.4\n%test\n")
            (result_dir / "calculation_report_manifest.json").write_text(json.dumps({"features": ["direct_pdf"]}), encoding="utf-8")
            (result_dir / "post_view.svg").write_text("<svg></svg>", encoding="utf-8")
            audit = audit_post_report_snapshot(result_dir=result_dir, output_dir=root / "audit")
            self.assertTrue(audit["summary"]["passed"])
            self.assertTrue(Path(audit["paths"]["json"]).exists())

    def test_result_table_page_limits_gui_rows_without_losing_source_rows(self) -> None:
        rows = [{"id": str(i), "value": str(i * 2)} for i in range(10)]
        first = result_table_page(rows, page_index=0, page_size=4)
        second = result_table_page(rows, page_index=1, page_size=4)
        last = result_table_page(rows, page_index=99, page_size=4)

        self.assertEqual(first.total_rows, 10)
        self.assertEqual(first.page_count, 3)
        self.assertEqual(len(first.rows), 4)
        self.assertEqual(second.rows[0]["id"], "4")
        self.assertEqual(last.page_index, 2)
        self.assertEqual(last.rows[-1]["id"], "9")
        self.assertEqual(first.headers, ["id", "value"])

    def test_result_table_rendering_uses_model_view_path(self) -> None:
        main_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        controller_source = Path("geofem_app/gui/post_report_controller.py").read_text(encoding="utf-8")
        render_body = controller_source.split("def _render_result_table_page", 1)[1].split("def _result_table_page_count", 1)[0]

        self.assertIn("class ResultTablePageModel(QAbstractTableModel)", main_source)
        self.assertIn("class ResultTableView(QTableView)", main_source)
        self.assertIn("self.result_table = ResultTableView()", main_source)
        self.assertIn("_set_result_table_model_page", render_body)
        self.assertNotIn("QTableWidgetItem(str(row_data.get(header", render_body)

    def test_csv_summary_and_page_read_stream_large_table_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "value", "label"])
                writer.writeheader()
                for i in range(11):
                    writer.writerow({"id": i, "value": i * 1.5, "label": f"row-{i}"})

            summary = summarize_csv_table(path)
            page = read_csv_table_page(path, page_index=2, page_size=4, total_rows=summary.row_count)

        self.assertEqual(summary.row_count, 11)
        self.assertEqual(summary.headers, ["id", "value", "label"])
        self.assertEqual(summary.numeric_fields, ["id", "value"])
        self.assertEqual(summary.minimums["value"], 0.0)
        self.assertEqual(summary.maximums["value"], 15.0)
        self.assertEqual(page.page_index, 2)
        self.assertEqual(page.start_row, 8)
        self.assertEqual(page.end_row, 11)
        self.assertEqual([row["id"] for row in page.rows], ["8", "9", "10"])

    def test_file_preview_truncates_display_and_guards_synchronous_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "huge.txt"
            text_path.write_text("0123456789" * 20, encoding="utf-8")
            json_path = root / "model.json"
            json_path.write_text('{"analysis": {"dimension": "2D"}}', encoding="utf-8")

            preview = preview_text_file(text_path, max_bytes=16)
            data, full_preview = read_mapping_file_guarded(json_path, max_bytes=128)
            parsed = read_json_file_guarded(json_path, max_bytes=128)

            with self.assertRaises(ValueError):
                read_mapping_file_guarded(json_path, max_bytes=4)
            with self.assertRaises(ValueError):
                read_json_file_guarded(json_path, max_bytes=4)

        self.assertTrue(preview.truncated)
        self.assertIn("GUI preview truncated", preview.text)
        self.assertEqual(data["analysis"]["dimension"], "2D")
        self.assertFalse(full_preview.truncated)
        self.assertEqual(parsed["analysis"]["dimension"], "2D")

    def test_audit_tail_and_recovery_metadata_avoid_full_table_read_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / ".geofem_audit_log.jsonl"
            with audit.open("w", encoding="utf-8") as f:
                for i in range(8):
                    f.write(json.dumps({"time": f"t{i}", "action": "a", "detail": {"i": i}}) + "\n")
            left = root / "left.yaml"
            right = root / "right.yaml"
            left.write_text("analysis:\n  dimension: 2D\nmesh:\n  nx: 1\n", encoding="utf-8")
            right.write_text("analysis:\n  dimension: 2D\nmesh:\n  nx: 2\n", encoding="utf-8")

            tail = read_jsonl_tail(audit, limit=3)
            candidates = recovery_candidate_infos([left, right], current_text=left.read_text(encoding="utf-8"))
            comparison = compare_recovery_files(left, right)

        self.assertEqual([row["detail"]["i"] for row in tail], [5, 6, 7])
        self.assertEqual(len(candidates), 2)
        self.assertTrue(any(not row["changed"] for row in candidates))
        self.assertTrue(comparison["ok"])
        self.assertEqual(comparison["mode"], "mapping")
        self.assertIn("mesh", comparison["changed"])


if __name__ == "__main__":
    unittest.main()
