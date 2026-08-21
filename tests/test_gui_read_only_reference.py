from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from geofem_app.samples import plane_strain_quad4_sample


class GuiReadOnlyReferenceTests(unittest.TestCase):
    def test_results_keep_inputs_navigable_but_read_only_until_reset(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QAbstractItemView
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
            original_table_triggers = window.material_table.editTriggers()

            with tempfile.TemporaryDirectory() as tmp:
                window.last_run_dir = Path(tmp)
                window.current_panel_key = "solver"
                window.panel_stack.setCurrentWidget(window.panel_pages["solver"])
                window._refresh_tree(select_panel="solver")
                window.refresh_workflow_guidance()
                window.refresh_solver_panel()
                QApplication.processEvents()

                root = window.tree.topLevelItem(0)
                geometry_item = window._find_tree_panel_item(root, "geometry")
                self.assertIsNotNone(geometry_item)
                self.assertFalse(geometry_item.isDisabled())
                self.assertTrue(window.workflow_ribbon_step_buttons[1].isEnabled())
                self.assertTrue(window.workflow_ribbon_reference_label.isVisible())
                self.assertTrue(window.analysis_reference_mode_active())
                self.assertFalse(window.autosave_timer.isActive())

                window._select_tree_panel("geometry")
                QApplication.processEvents()
                self.assertEqual(window.current_panel_key, "geometry")

                self.assertFalse(window.analysis_type.isEnabled())
                self.assertTrue(window.unit_system.isReadOnly())
                self.assertTrue(window.yaml_editor.isReadOnly())
                self.assertEqual(
                    window.material_table.editTriggers(),
                    QAbstractItemView.EditTrigger.NoEditTriggers,
                )
                self.assertFalse(window.primary_save_button.isEnabled())
                self.assertTrue(window.primary_run_button.isEnabled())
                self.assertTrue(window.primary_reset_results_button.isEnabled())

                cad_tools = {
                    str(button.property("cadToolLabel") or ""): button
                    for button in window.cad_tool_buttons
                }
                self.assertTrue(cad_tools["選択"].isEnabled())
                self.assertFalse(cad_tools["直線"].isEnabled())

                snapshot = window._cfg_snapshot()
                window.cfg.setdefault("analysis", {})["unit_system"] = "mutated"
                window._after_form_change("test mutation")
                self.assertEqual(window._cfg_snapshot(), snapshot)

                window.set_draw_mode("line")
                self.assertEqual(window.draw_mode, "select")

                window.reset_analysis_results()
                QApplication.processEvents()
                self.assertFalse(window.analysis_reference_mode_active())
                self.assertFalse(window.workflow_ribbon_reference_label.isVisible())
                self.assertTrue(window.analysis_type.isEnabled())
                self.assertFalse(window.unit_system.isReadOnly())
                self.assertFalse(window.yaml_editor.isReadOnly())
                self.assertEqual(window.material_table.editTriggers(), original_table_triggers)
                self.assertTrue(window.primary_save_button.isEnabled())
                self.assertTrue(cad_tools["直線"].isEnabled())
                self.assertTrue(window.autosave_timer.isActive())

            checks.append("reference-reset")
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, ["reference-reset"])


if __name__ == "__main__":
    unittest.main()
