from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_geofeas_post_report.py"


class GeoFEASPostReportAuditCliTests(unittest.TestCase):
    def test_post_report_audit_accepts_public_profile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result"
            stage = result / "Stage-1"
            out = root / "audit"
            stage.mkdir(parents=True)
            (result / "calculation_report.html").write_text("<html>report</html>", encoding="utf-8")
            (result / "calculation_report.pdf").write_bytes(b"%PDF-1.4\n%test\n")
            (result / "calculation_report_manifest.json").write_text(json.dumps({"features": ["direct_pdf", "geofeas_public_output_conditions"]}), encoding="utf-8")
            (result / "calculation_report_input_snapshot.json").write_text("{}", encoding="utf-8")
            (result / "geofeas_public_output_conditions.json").write_text(json.dumps({"save_behavior": {"commercial_oss_roundtrip": False}}), encoding="utf-8")
            (stage / "deformation.svg").write_text("<svg><path d='M0 0L1 1'/></svg>", encoding="utf-8")
            (stage / "displacements.csv").write_text("node_id,ux,uy\n1,0,0\n", encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--result", str(result), "--out", str(out), "--fail-on-error"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_post_report_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(compact["passed"])
            self.assertEqual(summary["error_count"], 0)
            self.assertGreaterEqual(summary["warning_count"], 1)
            self.assertEqual(summary["svg_count"], 1)
            self.assertEqual(summary["value_csv_count"], 1)
            self.assertFalse(summary["pixel_equivalent_post_claim"])
            self.assertTrue((out / "geofeas_post_report_audit.csv").exists())
            self.assertTrue((out / "geofeas_post_report_audit.html").exists())

    def test_post_report_audit_fails_when_required_outputs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result"
            out = root / "audit"
            result.mkdir()

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--result", str(result), "--out", str(out), "--fail-on-error"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_post_report_audit.json").read_text(encoding="utf-8"))
            self.assertFalse(compact["passed"])
            self.assertGreater(summary["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
