from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_geofeas_package.py"


class GeoFEASPackageInventoryCliTests(unittest.TestCase):
    def test_package_inventory_classifies_open_converter_and_private_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            out = root / "inventory"
            package.mkdir()
            (package / "model.GF2").write_bytes(b"GF2\x00private")
            (package / "condition.oss").write_bytes(b"OSS\x00private")
            (package / "mesh.sta").write_text("stage input", encoding="utf-8")
            (package / "drawing.dwg").write_bytes(b"AC1032\x00")
            (package / "shape.p21").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            (package / "results.csv").write_text("node_id,ux\n1,0.0\n", encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--package", str(package), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "geofeas_package_inventory.json").read_text(encoding="utf-8"))
            statuses = {row["relative_path"]: row["public_status"] for row in summary["rows"]}
            self.assertEqual(compact["file_count"], 6)
            self.assertEqual(summary["blocked_count"], 3)
            self.assertEqual(summary["converter_required_count"], 1)
            self.assertEqual(summary["open_supported_count"], 2)
            self.assertFalse(summary["native_private_roundtrip"])
            self.assertEqual(statuses["model.GF2"], "blocked_proprietary")
            self.assertEqual(statuses["drawing.dwg"], "converter_required")
            self.assertEqual(statuses["shape.p21"], "open_supported")
            self.assertTrue((out / "geofeas_package_inventory.csv").exists())
            self.assertTrue((out / "geofeas_package_inventory.html").exists())

    def test_package_inventory_can_fail_on_blocked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            out = root / "inventory"
            package.mkdir()
            (package / "model.GF2").write_bytes(b"GF2\x00private")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--package", str(package), "--out", str(out), "--fail-on-blocked"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            compact = json.loads(completed.stdout)
            self.assertEqual(compact["blocked_count"], 1)


if __name__ == "__main__":
    unittest.main()
