from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_geofeas_workflow_log.py"


class GeoFEASWorkflowLogCliTests(unittest.TestCase):
    def test_workflow_log_cli_accepts_public_tunnel_template(self) -> None:
        from geofem_app.geofeas_public import public_workflow_operation_log

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "operation_log.json"
            out = root / "verify"
            log_path.write_text(
                json.dumps({"operation_log": public_workflow_operation_log("tunnel_excavation")}, ensure_ascii=False),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--workflow", "tunnel_excavation", "--log", str(log_path), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_public_workflow_log_verification.json").read_text(encoding="utf-8"))
            self.assertTrue(compact["passed"])
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["expected_count"], summary["matched_count"])
            self.assertTrue((out / "geofeas_public_workflow_log_verification.csv").exists())
            self.assertTrue((out / "geofeas_public_workflow_log_verification.html").exists())

    def test_workflow_log_cli_rejects_missing_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "operation_log.csv"
            out = root / "verify"
            log_path.write_text("step,tab,action\n1,file,create a new public-profile model from the Tunnel.GF2 guidance scenario\n", encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--workflow", "tunnel_excavation", "--log", str(log_path), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_public_workflow_log_verification.json").read_text(encoding="utf-8"))
            self.assertFalse(compact["passed"])
            self.assertGreater(summary["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
