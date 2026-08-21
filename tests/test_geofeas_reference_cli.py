from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compare_geofeas_reference_package.py"


class GeoFEASReferencePackageCliTests(unittest.TestCase):
    def test_reference_package_cli_writes_json_csv_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "actual"
            reference = root / "reference"
            out = root / "compare"
            actual.mkdir()
            reference.mkdir()
            text = "node_id,ux,uy,u_norm,settlement\n1,0.0,-0.001,0.001,0.001\n"
            (actual / "displacements.csv").write_text(text, encoding="utf-8")
            (reference / "displacements.csv").write_text(text, encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--actual", str(actual), "--reference", str(reference), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_reference_package_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(compact["passed"])
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["schema"], "geofem.geofeas_reference_package_cli.v1")
            self.assertEqual(summary["recognized_reference_files"], ["displacements.csv"])
            self.assertEqual(summary["compared_count"], 1)
            self.assertTrue((out / "displacements_comparison.csv").exists())
            self.assertTrue((out / "displacements_tolerance.html").exists())
            self.assertTrue((out / "geofeas_package_tolerance.html").exists())

    def test_reference_package_cli_fails_empty_reference_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "actual"
            reference = root / "reference"
            out = root / "compare"
            actual.mkdir()
            reference.mkdir()

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--actual", str(actual), "--reference", str(reference), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            summary = json.loads((out / "geofeas_reference_package_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["passed"])
            self.assertTrue(summary["empty_reference_package"])
            self.assertIn("No recognized", summary["blocked_reason"])


if __name__ == "__main__":
    unittest.main()
