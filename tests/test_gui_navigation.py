from __future__ import annotations

import os
import unittest

from geofem_app.gui.workflow_panel import workflow_panel_layout_contract
from geofem_app.samples import plane_strain_quad4_sample


class GuiNavigationTests(unittest.TestCase):
    def test_navigation_surfaces_share_one_workspace_transition(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        import geofem_app.gui.main_window as main_window_module

        contract = workflow_panel_layout_contract()
        self.assertEqual(contract["canonical_navigation"], "workflow_ribbon")
        self.assertFalse(contract["diagnostic_navigation_aliases_visible"])

        original_exec = QApplication.exec
        checks: list[str] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[-1]
            window._load_cfg(plane_strain_quad4_sample(), keep_yaml=False)

            preview_calls: list[dict[str, object]] = []
            original_preview = window.update_preview
            window._model_preview_dirty = False
            window.update_preview = lambda **kwargs: preview_calls.append(dict(kwargs))
            window._activate_panel("geometry")
            self.assertEqual(len(preview_calls), 1)
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertEqual(window._last_navigation_source, "action")
            self.assertEqual(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole), "geometry")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            window.update_preview = original_preview

            mesh_item = window._navigation_tree_item("mesh")
            self.assertIsNotNone(mesh_item)
            window.tree.setCurrentItem(mesh_item)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "mesh")
            self.assertEqual(window._last_navigation_source, "tree")
            self.assertEqual(window.tree.currentItem(), mesh_item)

            materials_index = next(
                index
                for index, step in enumerate(window.workflow_guidance["steps"])
                if step.get("panel") == "materials"
            )
            window.workflow_ribbon_step_buttons[materials_index].click()
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "materials")
            self.assertEqual(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole), "materials")
            self.assertEqual(window._last_navigation_source, "action")

            root = window.tree.topLevelItem(0)
            self.assertEqual(root.data(0, Qt.ItemDataRole.UserRole), "workflow")
            window.tree.setCurrentItem(root)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "workflow")
            self.assertIs(window.tabs.currentWidget(), window.panel_stack)
            self.assertIs(window.panel_stack.currentWidget(), window.panel_pages["workflow"])
            self.assertTrue(window.workflow_back_button.isHidden())
            self.assertTrue(window.workflow_jump_button.isHidden())

            window._activate_panel("geometry")
            geometry_item = window.tree.currentItem()
            original_blocked = window._navigation_target_blocked
            window._navigation_target_blocked = lambda raw_key, **_kwargs: window._panel_key_from_tree_key(raw_key) == "mesh"
            window.tree.setCurrentItem(mesh_item)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "geometry")
            self.assertIs(window.tree.currentItem(), geometry_item)
            window._navigation_target_blocked = original_blocked

            checks.append("action-tree-ribbon-overview-blocked")
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, ["action-tree-ribbon-overview-blocked"])


if __name__ == "__main__":
    unittest.main()
