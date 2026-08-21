from __future__ import annotations

import os
import unittest


class GuiWorkspaceFocusModeTests(unittest.TestCase):
    def test_model_and_result_focus_mode_preserves_and_restores_workspace(self) -> None:
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
            window.resize(1700, 900)
            window._activate_panel("geometry")
            window._set_log_drawer_expanded(True)
            QApplication.processEvents()

            self.assertTrue(window.workflow_ribbon_focus_button.isVisible())
            self.assertTrue(window.workflow_ribbon_focus_button.isEnabled())
            self.assertFalse(window.workflow_ribbon_focus_button.icon().isNull())
            self.assertEqual(window.view_focus_action.shortcut().toString(), "F11")
            self.assertFalse(window.workspace_focus_mode_active())
            self.assertTrue(window.left_navigation_panel.isVisible())
            self.assertTrue(window.right_panel_stack.isVisible())
            self.assertTrue(window.bottom_panel.isVisible())
            self.assertTrue(window.log.isVisible())
            original_workspace_sizes = list(window.workspace_splitter.sizes())
            original_main_sizes = list(window.main_splitter.sizes())

            window.workflow_ribbon_focus_button.click()
            QApplication.processEvents()

            self.assertTrue(window.workspace_focus_mode_active())
            self.assertTrue(window.workflow_ribbon_focus_button.isChecked())
            self.assertTrue(window.view_focus_action.isChecked())
            self.assertEqual(window.workflow_ribbon_focus_button.text(), "通常表示")
            self.assertTrue(window.left_navigation_panel.isHidden())
            self.assertTrue(window.right_panel_stack.isHidden())
            self.assertTrue(window.bottom_panel.isHidden())
            self.assertTrue(window.log.isHidden())
            self.assertTrue(window.workflow_ribbon_step_container.isHidden())
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)

            window._activate_panel("mesh")
            QApplication.processEvents()
            self.assertTrue(window.workspace_focus_mode_active())
            self.assertEqual(window.current_panel_key, "mesh")
            self.assertTrue(window.left_navigation_panel.isHidden())

            window._activate_panel("results")
            QApplication.processEvents()
            self.assertTrue(window.workspace_focus_mode_active())
            self.assertEqual(window.current_panel_key, "results")
            self.assertTrue(window.right_panel_stack.isHidden())

            window.view_focus_action.trigger()
            QApplication.processEvents()
            self.assertFalse(window.workspace_focus_mode_active())
            self.assertFalse(window.workflow_ribbon_focus_button.isChecked())
            self.assertTrue(window.left_navigation_panel.isVisible())
            self.assertTrue(window.right_panel_stack.isVisible())
            self.assertTrue(window.bottom_panel.isVisible())
            self.assertTrue(window.log.isVisible())
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(original_workspace_sizes, window.workspace_splitter.sizes())),
                3,
            )
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(original_main_sizes, window.main_splitter.sizes())),
                3,
            )

            window.workflow_ribbon_focus_button.click()
            QApplication.processEvents()
            self.assertTrue(window.workspace_focus_mode_active())
            window._activate_panel("analysis")
            QApplication.processEvents()
            self.assertFalse(window.workspace_focus_mode_active())
            self.assertFalse(window.workflow_ribbon_focus_button.isEnabled())
            self.assertTrue(window.left_navigation_panel.isVisible())
            self.assertTrue(window.right_panel_stack.isVisible())

            window._activate_panel("geometry")
            window.workflow_ribbon_focus_button.click()
            QApplication.processEvents()
            self.assertTrue(window.workspace_focus_mode_active())
            window._set_solver_navigation_suppressed(True)
            QApplication.processEvents()
            self.assertFalse(window.workspace_focus_mode_active())
            self.assertFalse(window.workflow_ribbon_focus_button.isEnabled())
            window._set_solver_navigation_suppressed(False)
            QApplication.processEvents()
            self.assertTrue(window.workflow_ribbon_focus_button.isEnabled())

            checks.append("focus-restored")
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, ["focus-restored"])


if __name__ == "__main__":
    unittest.main()
