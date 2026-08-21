from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_geofeas_material_profile.py"


class GeoFEASMaterialAuditCliTests(unittest.TestCase):
    def test_material_audit_reports_public_substitutes_and_private_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "model.json"
            out = root / "audit"
            cfg.write_text(
                json.dumps(
                    {
                        "materials": {
                            "soil": {"model": "elastic", "E": 10000.0, "nu": 0.3},
                            "sand": {
                                "model": "bilinear_liquefaction",
                                "G0": 25000.0,
                                "gamma_ref": 0.001,
                                "liquefaction": {"cyclic_resistance_ratio": 0.2, "cyclic_stress_ratio": 0.18},
                            },
                            "pz": {"model": "pastor_zienkiewicz_sand", "G0": 20000.0, "gamma_ref": 0.002, "friction_angle": 32.0},
                        },
                        "steps": [{"name": "liq", "geofeas_workflow": "river_liquefaction_h28"}],
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--config", str(cfg), "--out", str(out), "--fail-on-error"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_public_material_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(compact["passed"])
            self.assertEqual(summary["material_count"], 3)
            self.assertEqual(summary["public_substitute_count"], 2)
            self.assertEqual(summary["open_supported_count"], 1)
            self.assertFalse(summary["native_geo_feas_material_equivalence"])
            self.assertTrue(any(row["model"] == "bilinear_liquefaction" and "Exact ru" in row["geofeas_private_gap"] for row in summary["rows"]))
            self.assertTrue((out / "geofeas_public_material_audit.csv").exists())
            self.assertTrue((out / "geofeas_public_material_audit.html").exists())

    def test_material_audit_fails_unknown_or_incomplete_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "model.json"
            out = root / "audit"
            cfg.write_text(json.dumps({"materials": {"bad": {"model": "pastor_zienkiewicz_clay", "G0": 1000.0}}}), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--config", str(cfg), "--out", str(out), "--fail-on-error"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            summary = json.loads((out / "geofeas_public_material_audit.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["error_count"], 1)
            self.assertIn("gamma_ref", summary["rows"][0]["missing_required"])


if __name__ == "__main__":
    unittest.main()
