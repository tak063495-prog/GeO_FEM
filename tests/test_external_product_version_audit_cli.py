from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_external_product_versions.py"


class ExternalProductVersionAuditCliTests(unittest.TestCase):
    def test_external_product_audit_reports_products_versions_and_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            out = root / "audit"
            package.mkdir()
            (package / "VGFlow_ver7_heads.csv").write_text("time,node_id,head,extra_col\n0,1,2.0,x\n", encoding="utf-8")
            (package / "UC-1_pressure.tsv").write_text("時刻\t節点番号\t水圧\n0\t1\t10.0\n", encoding="utf-8")
            (package / "GeoFEAS_model.GF2").write_bytes(b"GF2\x00private")
            (package / "drawing.dwg").write_bytes(b"AC1032\x00")

            completed = subprocess.run(
                [sys.executable, str(TOOL), "--package", str(package), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            summary = json.loads((out / "external_product_version_audit.json").read_text(encoding="utf-8"))
            rows = {row["relative_path"]: row for row in summary["rows"]}
            self.assertEqual(compact["file_count"], 4)
            self.assertEqual(summary["blocked_count"], 1)
            self.assertEqual(summary["converter_required_count"], 1)
            self.assertEqual(summary["open_supported_count"], 2)
            self.assertFalse(summary["exact_product_version_parity"])
            self.assertEqual(rows["VGFlow_ver7_heads.csv"]["detected_product"], "VGFlow")
            self.assertEqual(rows["VGFlow_ver7_heads.csv"]["field_mapping_status"], "partial_headers_mapped")
            self.assertEqual(rows["UC-1_pressure.tsv"]["detected_product"], "UC-1")
            self.assertEqual(rows["UC-1_pressure.tsv"]["field_mapping_status"], "all_headers_mapped")
            self.assertEqual(rows["GeoFEAS_model.GF2"]["public_status"], "blocked_proprietary")
            self.assertTrue((out / "external_product_version_audit.csv").exists())
            self.assertTrue((out / "external_product_version_audit.html").exists())

    def test_external_product_audit_can_fail_on_blocked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            out = root / "audit"
            package.mkdir()
            (package / "GeoFEAS_model.GF2").write_bytes(b"GF2\x00private")

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
