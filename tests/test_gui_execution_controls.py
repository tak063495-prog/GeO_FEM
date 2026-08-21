from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from geofem_app.samples import plane_strain_quad4_sample


class GuiExecutionControlTests(unittest.TestCase):
    def test_run_stop_reset_and_save_controls_follow_one_solver_state(self) -> None:
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

            run_controls = [
                window.primary_run_button,
                window.aux_run_button,
                window.aux_solver_run_button,
                window.analysis_run_action,
                window.command_actions["analysis.run"],
            ]
            stop_controls = [
                window.primary_stop_button,
                window.aux_solver_stop_button,
                window.analysis_stop_action,
                window.command_actions["analysis.stop"],
            ]
            reset_controls = [
                window.primary_reset_results_button,
                window.aux_solver_reset_button,
                window.analysis_reset_results_action,
            ]
            save_controls = [
                window.primary_save_button,
                window.aux_solver_save_button,
                window.file_actions["save"],
                window.file_actions["save_as"],
                window.command_actions["input.save"],
            ]

            def assert_enabled(controls: list[object], expected: bool) -> None:
                self.assertTrue(controls)
                self.assertTrue(all(control.isEnabled() is expected for control in controls))

            window.last_run_dir = None
            window._loaded_result_summary_path = None
            state = window._refresh_execution_action_states([])
            self.assertEqual(state["phase"], "idle")
            assert_enabled(run_controls, True)
            assert_enabled(stop_controls, False)
            assert_enabled(reset_controls, False)
            assert_enabled(save_controls, True)

            with tempfile.TemporaryDirectory() as tmp:
                window.last_run_dir = Path(tmp)
                state = window._refresh_execution_action_states([])
                self.assertEqual(state["phase"], "results")
                assert_enabled(run_controls, True)
                assert_enabled(stop_controls, False)
                assert_enabled(reset_controls, True)
                assert_enabled(save_controls, False)

                window._solver_preflight_job_id = "model_check:test"
                window.refresh_solver_panel()
                self.assertIn("直前チェック中", window.solver_status_label.text())
                assert_enabled(run_controls, False)
                assert_enabled(stop_controls, False)
                assert_enabled(reset_controls, False)
                assert_enabled(save_controls, False)

                window._solver_preflight_job_id = ""
                original_running = window._solver_process_running
                window._solver_process_running = lambda: True
                job_id = window.gui_jobs.start_job("solver", target="test")
                window._solver_job_id = job_id
                window.refresh_solver_panel()
                self.assertIn("実行中", window.solver_status_label.text())
                assert_enabled(run_controls, False)
                assert_enabled(stop_controls, True)
                assert_enabled(reset_controls, False)
                assert_enabled(save_controls, False)

                cancel_file_requests: list[bool] = []
                original_cancel_file_request = window._request_solver_cancel_file
                window._request_solver_cancel_file = lambda: cancel_file_requests.append(True)
                window.stop_solver()
                self.assertIn("中断要求済み", window.solver_status_label.text())
                assert_enabled(run_controls, False)
                assert_enabled(stop_controls, False)
                assert_enabled(reset_controls, False)
                assert_enabled(save_controls, False)
                self.assertTrue(all(control.property("executionState") == "cancel_requested" for control in run_controls))
                window.stop_solver()
                self.assertEqual(cancel_file_requests, [True])
                window._request_solver_cancel_file = original_cancel_file_request

                window.gui_jobs.cancel_job(job_id, message="test cleanup")
                window._solver_job_id = ""
                window._solver_cancel_requested_local = False
                window._solver_process_running = original_running

            checks.append("idle-results-preflight-running-cancel")
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(main_window_module.run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(checks, ["idle-results-preflight-running-cancel"])


if __name__ == "__main__":
    unittest.main()
