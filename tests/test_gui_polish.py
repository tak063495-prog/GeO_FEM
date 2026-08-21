from __future__ import annotations

import os
from pathlib import Path
import json
import sys
import tempfile
import time
import unittest

from geofem_app.gui.desktop_layout import resolve_desktop_layout_profile
from geofem_app.gui.design_system import commercial_gui_stylesheet
from geofem_app.gui.project_paths import resolve_initial_project_root
from geofem_app.gui.template_catalog import clear_input_template_metadata_cache, read_input_template_metadata


class GuiPolishTests(unittest.TestCase):
    def test_initial_project_root_avoids_install_location_and_keeps_writable_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install = root / "Program Files" / "WindowsApps" / "GeoFEM"
            install.mkdir(parents=True)
            home = root / "user"
            fallback = resolve_initial_project_root(install, home=home, environ={})
            self.assertEqual(fallback, (home / "Documents" / "GeoFEM").resolve())

            project = root / "work" / "project"
            selected = resolve_initial_project_root(project, home=home, environ={})
            self.assertEqual(selected, project.resolve())

    def test_template_catalog_reads_metadata_without_full_config_expansion(self) -> None:
        clear_input_template_metadata_cache()
        path = Path("examples/slope_srm_drucker_prager.yaml")
        started = time.perf_counter()
        metadata = read_input_template_metadata(path)
        elapsed = time.perf_counter() - started
        self.assertEqual(metadata.analysis_type, "static_plane_strain")
        self.assertEqual(metadata.element_type, "QUAD4")
        self.assertEqual(metadata.integration, "B-bar")
        self.assertEqual(metadata.integration_variants, ("FULL", "B-bar", "SRI"))
        self.assertLess(elapsed, 0.5)
        self.assertIs(read_input_template_metadata(path), metadata)

    def test_1024_layout_has_consistent_minimums(self) -> None:
        profile = resolve_desktop_layout_profile(1024, 700)
        self.assertEqual(profile.minimum_window_size, (1024, 700))
        self.assertEqual(sum(profile.horizontal_split_sizes), profile.window_width)
        self.assertLessEqual(sum((profile.tree_min_width, profile.center_min_width, profile.panel_min_width)), profile.window_width)
        self.assertGreaterEqual(profile.model_view_min_size, (360, 260))
        stylesheet = commercial_gui_stylesheet()
        self.assertIn("QPushButton:focus", stylesheet)
        self.assertIn("QPushButton:disabled", stylesheet)

    def test_small_window_scrolls_auxiliary_panel_and_shares_log_document(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QScrollArea
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        original_resolver = main_window_module.resolve_desktop_layout_profile
        checks: list[bool] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            window.resize(1024, 700)
            window._select_tree_panel("geometry")
            QApplication.processEvents()

            self.assertIsInstance(window.auxiliary_panel, QScrollArea)
            self.assertIs(window.auxiliary_panel.widget(), window.auxiliary_panel_content)
            self.assertGreater(window.auxiliary_panel.viewport().height(), 0)
            self.assertGreater(window.auxiliary_panel.verticalScrollBar().maximum(), 0)
            visible_fields = [
                field
                for field in (
                    *window.aux_geometry_settings_group.findChildren(QLineEdit),
                    *window.aux_geometry_settings_group.findChildren(QComboBox),
                )
                if field.isVisible()
            ]
            self.assertTrue(visible_fields)
            self.assertTrue(all(field.height() >= 24 for field in visible_fields))
            self.assertTrue(all(size > 0 for size in window.workspace_splitter.sizes()))

            self.assertIs(window.log.document(), window.solver_log.document())
            window.append_log("[GUI] shared-log-check")
            window._flush_log_buffer()
            self.assertIn("shared-log-check", window.solver_log.toPlainText())
            self.assertFalse(window.log.isVisible())
            window.global_log_toggle_button.click()
            QApplication.processEvents()
            self.assertTrue(window.log.isVisible())

            window.refresh_workflow_guidance()
            status_lines = window._auxiliary_status_lines("analysis")
            self.assertTrue(any("任意設定" in line for line in status_lines))
            self.assertFalse(any("未入力/不足" in line for line in status_lines))

            window._select_tree_panel("solver")
            QApplication.processEvents()
            self.assertTrue(window.bottom_panel.isHidden())
            self.assertTrue(window.solver_log.isVisible())

            manifests: list[str] = []
            window._write_gui_worker_failure_manifest = lambda _job, _message, context=None: manifests.append(_message) or None
            for _index in range(2):
                job_id = window.gui_jobs.start_job("autosave", target="autosave")
                window._autosave_job_id = job_id
                window._autosave_failed(job_id, "permission denied")
            self.assertEqual(manifests, ["permission denied"])
            self.assertEqual(window._autosave_failure_count, 2)
            self.assertGreater(window._autosave_next_retry_monotonic, time.monotonic())

            checks.append(True)
            window.close()
            return 0

        QApplication.exec = fake_exec
        main_window_module.resolve_desktop_layout_profile = lambda _width, _height: original_resolver(1024, 700)
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
            main_window_module.resolve_desktop_layout_profile = original_resolver
        self.assertEqual(checks, [True])

    def test_phase_one_to_three_gui_surfaces_are_responsive_and_stateful(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        original_resolver = main_window_module.resolve_desktop_layout_profile
        checks: list[bool] = []

        def fake_exec(app: QApplication) -> int:
            window = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"][-1]
            window.resize(1366, 768)
            QApplication.processEvents()

            self.assertTrue(window.analysis_axisym_box.isHidden())
            self.assertTrue(window.analysis_output_custom_row.isHidden())
            custom_index = window.analysis_output_policy.findData("custom")
            window.analysis_output_policy.setCurrentIndex(custom_index)
            QApplication.processEvents()
            self.assertTrue(window.analysis_output_custom_row.isVisible())
            window.analysis_geometry.setCurrentText("axisymmetric")
            QApplication.processEvents()
            self.assertTrue(window.analysis_axisym_box.isVisible())
            self.assertIs(window.analysis_field_labels["analysis_type"].buddy(), window.analysis_type)

            self.assertEqual(window.material_tabs.count(), 2)
            self.assertTrue(all(window.material_table.isColumnHidden(column) for column in range(7, 14)))
            window.material_advanced_columns_button.click()
            self.assertTrue(all(not window.material_table.isColumnHidden(column) for column in range(7, 14)))

            window._activate_panel("results")
            QApplication.processEvents()
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertEqual(window.results_tabs.count(), 4)
            self.assertIn("解析結果がまだありません", window.results_summary.text())
            self.assertTrue(all(not button.isEnabled() for button in window.result_data_buttons))
            self.assertTrue(window.result_empty_state.isVisible())
            self.assertFalse(window.result_judgment_panel.isVisible())
            self.assertFalse(window.result_primary_actions_widget.isVisible())
            self.assertFalse(window.result_more_button.isVisible())
            self.assertFalse(window.results_tabs.isTabEnabled(1))
            self.assertFalse(window.results_tabs.isTabEnabled(2))

            window._select_tree_panel("analysis")
            QApplication.processEvents()
            window.workspace_splitter.setSizes([310, 720, 336])
            QApplication.processEvents()
            remembered = list(window.workspace_splitter.sizes())
            window.set_left_navigation_visible(False)
            self.assertTrue(window.left_navigation_panel.isHidden())
            self.assertFalse(window.view_navigation_action.isChecked())
            window.set_left_navigation_visible(True)
            QApplication.processEvents()
            restored = list(window.workspace_splitter.sizes())
            self.assertTrue(window.left_navigation_panel.isVisible())
            self.assertTrue(window.view_navigation_action.isChecked())
            self.assertLessEqual(max(abs(a - b) for a, b in zip(remembered, restored)), 3)

            self.assertTrue(window.workflow_ribbon_step_container.isHidden())
            self.assertLessEqual(window.workflow_ribbon.maximumHeight(), 52)
            window.resize(1700, 900)
            QApplication.processEvents()
            self.assertTrue(window.workflow_ribbon_step_container.isVisible())
            self.assertGreaterEqual(window.workflow_ribbon.minimumHeight(), 104)

            window._set_log_drawer_expanded(True)
            QApplication.processEvents()
            self.assertTrue(window.log.isVisible())
            self.assertTrue(window.global_log_toggle_button.isChecked())
            self.assertTrue(window.view_log_action.isChecked())
            window.view_log_action.trigger()
            QApplication.processEvents()
            self.assertFalse(window.log.isVisible())
            self.assertFalse(window.global_log_toggle_button.isChecked())

            checks.append(True)
            window.close()
            return 0

        QApplication.exec = fake_exec
        main_window_module.resolve_desktop_layout_profile = lambda _width, _height: original_resolver(1366, 768)
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
            main_window_module.resolve_desktop_layout_profile = original_resolver
        self.assertEqual(checks, [True])

    def test_numba_warmup_uses_short_lived_process(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        original_exec = QApplication.exec
        original_enabled = main_window_module.gui_numba_warmup_enabled
        captured: list[dict[str, object]] = []
        child_summary = {
            "schema": "geofem.numba_warmup.v1",
            "enabled": True,
            "profile": "gui",
            "elapsed_seconds": 0.01,
            "kernel_count": 1,
            "warmed_count": 1,
            "failed_count": 0,
            "kernels": [],
        }

        def fake_exec(app: QApplication) -> int:
            window = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"][-1]
            payload = json.dumps(child_summary, separators=(",", ":"))
            window._numba_warmup_command = lambda: (sys.executable, ["-c", f"print({payload!r})"])
            window.start_numba_warmup_async()
            deadline = time.monotonic() + 5.0
            while window._numba_warmup_job_id and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.01)
            self.assertEqual(window._numba_warmup_job_id, "")
            self.assertIsNone(window._numba_warmup_process)
            captured.append(dict(window.numba_warmup_summary))
            window.close()
            return 0

        QApplication.exec = fake_exec
        main_window_module.gui_numba_warmup_enabled = lambda: True
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec
            main_window_module.gui_numba_warmup_enabled = original_enabled
        self.assertEqual(captured, [child_summary])


if __name__ == "__main__":
    unittest.main()
