from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_geofeas_public_acceptance.py"


class GeoFEASPublicAcceptanceCliTests(unittest.TestCase):
    def test_public_acceptance_runner_consolidates_available_audits(self) -> None:
        from geofem_app.geofeas_public import public_workflow_operation_log

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "model.json"
            result = root / "result"
            stage = result / "Stage-1"
            package = root / "package"
            operation_log = root / "operation_log.json"
            out = root / "acceptance"
            stage.mkdir(parents=True)
            package.mkdir()
            cfg.write_text(json.dumps({"materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.3}}}), encoding="utf-8")
            (result / "calculation_report.html").write_text("<html>report</html>", encoding="utf-8")
            (result / "calculation_report.pdf").write_bytes(b"%PDF-1.4\n%test\n")
            (result / "calculation_report_manifest.json").write_text(json.dumps({"features": ["direct_pdf"]}), encoding="utf-8")
            (result / "calculation_report_input_snapshot.json").write_text("{}", encoding="utf-8")
            (result / "geofeas_public_output_conditions.json").write_text(json.dumps({"save_behavior": {"commercial_oss_roundtrip": False}}), encoding="utf-8")
            (stage / "deformation.svg").write_text("<svg><path d='M0 0L1 1'/></svg>", encoding="utf-8")
            (stage / "displacements.csv").write_text("node_id,ux,uy\n1,0,0\n", encoding="utf-8")
            (package / "shape.p21").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            (package / "VGFlow_heads.csv").write_text("time,node_id,head\n0,1,2.0\n", encoding="utf-8")
            operation_log.write_text(json.dumps({"operation_log": public_workflow_operation_log("tunnel_excavation")}), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--out",
                    str(out),
                    "--config",
                    str(cfg),
                    "--result",
                    str(result),
                    "--package",
                    str(package),
                    "--operation-log",
                    str(operation_log),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_public_acceptance_summary.json").read_text(encoding="utf-8"))
            modules = {row["module"]: row for row in summary["modules"]}
            self.assertTrue(compact["passed"])
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["skipped_count"], 1)
            self.assertFalse(summary["native_geo_feas_equivalence_claim"])
            self.assertEqual(modules["material_profile"]["status"], "passed")
            self.assertEqual(modules["post_report"]["status"], "passed")
            self.assertEqual(modules["package_inventory"]["status"], "passed")
            self.assertEqual(modules["external_product_versions"]["status"], "passed")
            self.assertEqual(modules["workflow_log"]["status"], "passed")
            self.assertEqual(modules["reference_comparison"]["status"], "skipped")
            self.assertTrue((out / "geofeas_public_acceptance_summary.csv").exists())
            self.assertTrue((out / "geofeas_public_acceptance_summary.html").exists())

    def test_public_acceptance_runner_can_require_all_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "acceptance"
            completed = subprocess.run(
                [sys.executable, str(TOOL), "--out", str(out), "--require-all"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_public_acceptance_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(compact["passed"])
            self.assertEqual(summary["skipped_count"], 6)
            self.assertGreater(summary["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
