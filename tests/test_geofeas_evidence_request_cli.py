from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "create_geofeas_evidence_request.py"


class GeoFEASEvidenceRequestCliTests(unittest.TestCase):
    def test_evidence_request_pack_writes_manifest_tables_and_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence_request"
            completed = subprocess.run(
                [sys.executable, str(TOOL), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            compact = json.loads(completed.stdout)
            manifest = json.loads((out / "geofeas_evidence_request.json").read_text(encoding="utf-8"))
            self.assertEqual(compact["item_count"], 6)
            self.assertFalse(compact["native_geo_feas_equivalence_claim"])
            self.assertEqual(manifest["schema"], "geofem.geofeas_evidence_request_pack.v1")
            self.assertEqual(manifest["acceptance_gate"], "tools/run_geofeas_public_acceptance.py")
            self.assertTrue((out / "geofeas_evidence_request.csv").exists())
            self.assertTrue((out / "geofeas_evidence_request.html").exists())
            self.assertTrue((out / "README.md").exists())
            for item_id in ["E1", "E2", "E3", "E4", "E5", "E6"]:
                self.assertTrue((out / "evidence" / item_id / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
