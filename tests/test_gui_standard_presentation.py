from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from geofem_app.gui.presentation_labels import friendly_input_reference, friendly_stage_name


class GuiStandardPresentationTests(unittest.TestCase):
    def test_input_references_are_presented_without_changing_unknown_text(self) -> None:
        self.assertNotEqual(friendly_input_reference("analysis.type"), "analysis.type")
        self.assertNotEqual(friendly_input_reference("materials.soil.E"), "materials.soil.E")
        self.assertEqual(friendly_input_reference("results/summary.json"), "解析結果サマリ")
        self.assertNotIn("_", friendly_stage_name("case4_srm_strength_reduction"))
        self.assertEqual(friendly_input_reference("任意の説明文"), "任意の説明文")

    def test_standard_mode_hides_internal_values_and_keeps_storage_ids(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QMessageBox
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[bool] = []

        def fake_exec(app: QApplication) -> int:
            window = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"][-1]
            QApplication.processEvents()

            self.assertEqual(window._current_gui_operation_mode(), "standard")
            self.assertEqual(window.analysis_type.currentData(), "static_plane_strain")
            self.assertNotEqual(window.analysis_type.currentText(), "static_plane_strain")
            window.analysis_type.setCurrentText("srm")
            self.assertEqual(window.analysis_type.currentData(), "srm")
            window.apply_analysis_panel()
            self.assertEqual(window.cfg["analysis"]["type"], "srm")

            window._activate_panel("stages")
            QApplication.processEvents()
            self.assertTrue(window.left_command_tools_widget.isHidden())
            self.assertFalse(window.stage_standard_action_bar.isHidden())
            self.assertTrue(window.stage_change_actions_widget.isHidden())
            self.assertTrue(window.stage_change_table.isHidden())
            self.assertTrue(window.stage_standard_remove_change_button.isHidden())
            self.assertTrue(window.stage_standard_apply_changes_button.isHidden())
            internal_index = window.stage_workspace_tabs.indexOf(window.stage_internal_data_tab)
            self.assertGreaterEqual(internal_index, 0)
            self.assertFalse(window.stage_workspace_tabs.isTabVisible(internal_index))
            solver_wrappers = [
                wrapper
                for wrapper in window.stage_inspector_field_wrappers
                if str(wrapper.property("stageInspectorFamily") or "") == "solver"
            ]
            self.assertTrue(solver_wrappers)
            self.assertTrue(all(wrapper.isHidden() for wrapper in solver_wrappers))

            window.yaml_editor.setPlainText("analysis:\n  type: srm\n")
            window.refresh_model_check_overview()
            self.assertIn("正常", window.model_check_yaml_status.text())
            self.assertNotIn("keys:", window.model_check_yaml_status.text())
            main_window_module.apply_model_check_issues_view(
                window,
                [("ERROR", "analysis.type", "解析種別を確認してください。", {})],
                window._model_check_panel_qt(),
            )
            target_item = window.check_table.item(0, 1)
            self.assertNotEqual(target_item.text(), "analysis.type")
            self.assertEqual(target_item.data(Qt.ItemDataRole.UserRole)["_raw_target"], "analysis.type")

            detail_index = window.gui_operation_mode_combo.findData("detail")
            window.gui_operation_mode_combo.setCurrentIndex(detail_index)
            QApplication.processEvents()
            self.assertFalse(window.left_command_tools_widget.isHidden())
            self.assertTrue(window.stage_standard_action_bar.isHidden())
            self.assertFalse(window.stage_change_actions_widget.isHidden())
            self.assertTrue(window.stage_workspace_tabs.isTabVisible(internal_index))
            main_window_module.apply_model_check_issues_view(
                window,
                [("ERROR", "analysis.type", "解析種別を確認してください。", {})],
                window._model_check_panel_qt(),
            )
            self.assertEqual(window.check_table.item(0, 1).text(), "analysis.type")
            window.refresh_model_check_overview()
            self.assertIn("keys:", window.model_check_yaml_status.text())

            window._apply_yaml_load_navigation_choice(QMessageBox.StandardButton.Yes, source_label="case.yaml")
            self.assertEqual(window.current_panel_key, "solver")
            window._apply_yaml_load_navigation_choice(QMessageBox.StandardButton.No, source_label="case.yaml")
            self.assertEqual(window.current_panel_key, "analysis")

            checks.append(True)
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [True])

    def test_external_result_navigation_empty_state_notifications_and_viewports(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        checks: list[bool] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "external_run"
            results_dir = run_dir / "results"
            stage_dir = results_dir / "case4_srm_strength_reduction"
            stage_dir.mkdir(parents=True)
            summary_path = results_dir / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "analysis": "srm",
                        "stages": [
                            {
                                "name": "case4_srm_strength_reduction",
                                "output_dir": str(stage_dir),
                                "max_displacement": 0.05,
                                "solver": {
                                    "converged": True,
                                    "performance": {"elapsed_seconds": 12.5},
                                    "srm": {
                                        "factor_of_safety": 2.1,
                                        "stable_factor": 2.1,
                                        "failed_factor": 2.105,
                                        "factor_tol": 0.005,
                                        "search_mode": "explicit_factors",
                                        "trials": [{"factor": 2.1}, {"factor": 2.105}],
                                    },
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_exec(app: QApplication) -> int:
                window = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"][-1]
                QApplication.processEvents()

                window.last_run_dir = None
                window._loaded_result_summary_path = None
                window._activate_panel("results")
                window._refresh_result_action_states()
                QApplication.processEvents()
                self.assertTrue(window.result_primary_actions_widget.isHidden())
                self.assertTrue(window.result_more_button.isHidden())
                self.assertTrue(window.aux_result_control_group.isHidden())
                self.assertTrue(window.result_empty_state.isVisible())
                for index in range(1, min(4, window.results_tabs.count())):
                    self.assertFalse(window.results_tabs.isTabVisible(index))

                window.focus_result_display_controls()
                QApplication.processEvents()
                self.assertTrue(window.notification_banner.isVisible())
                self.assertIn("解析結果", window.notification_banner.text())

                window._load_result_summary(summary_path)
                window._activate_panel("results")
                QApplication.processEvents()
                self.assertIs(window.tabs.currentWidget(), window.panel_stack)
                self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["results"])
                self.assertEqual(window._current_summary_path().resolve(), summary_path.resolve())
                self.assertEqual(window._active_result_dir().resolve(), results_dir.resolve())
                result_step = next(row for row in window.workflow_guidance["steps"] if row["id"] == "results")
                self.assertTrue(result_step["completed"])
                self.assertFalse(window.result_primary_actions_widget.isHidden())
                self.assertFalse(window.aux_result_control_group.isHidden())
                self.assertTrue(window.aux_selection_context_group.isHidden())
                self.assertNotIn("explicit_factors", window.result_judgment_detail.text())
                self.assertNotIn("case4_srm_strength_reduction", window.result_judgment_detail.text())
                self.assertNotIn("_", window.result_stage_selector.currentText())

                icon_keys = {
                    window.result_primary_visual_button.icon().cacheKey(),
                    window.result_primary_report_button.icon().cacheKey(),
                    window.result_primary_folder_button.icon().cacheKey(),
                    window.result_primary_srm_button.icon().cacheKey(),
                }
                self.assertEqual(len(icon_keys), 4)

                window.notify_user("保存しました", severity="success", timeout_ms=0)
                self.assertEqual(window.notification_banner.property("severity"), "ok")
                self.assertEqual(window.notification_banner.text(), "保存しました")

                window._activate_panel("stages")
                for width, height in ((1366, 768), (1440, 900)):
                    window.resize(width, height)
                    QApplication.processEvents()
                    self.assertGreater(window.workspace_splitter.width(), 0)
                    self.assertEqual(len(window.workspace_splitter.sizes()), 3)
                    self.assertTrue(all(size > 0 for size in window.workspace_splitter.sizes()))
                    for button in (
                        window.stage_standard_add_stage_button,
                        window.stage_standard_manage_button,
                        window.stage_standard_add_change_button,
                        window.stage_standard_remove_change_button,
                        window.stage_standard_apply_changes_button,
                        window.stage_standard_open_detail_button,
                    ):
                        if button.isHidden():
                            continue
                        text_width = button.fontMetrics().horizontalAdvance(button.text())
                        icon_width = button.iconSize().width() + 10 if not button.icon().isNull() else 0
                        self.assertLessEqual(text_width + icon_width + 18, button.width())

                checks.append(True)
                window.close()
                return 0

            QApplication.exec = fake_exec
            try:
                self.assertEqual(main_window_module.run_gui(), 0)
            finally:
                QApplication.exec = original_exec
        self.assertEqual(checks, [True])


if __name__ == "__main__":
    unittest.main()
