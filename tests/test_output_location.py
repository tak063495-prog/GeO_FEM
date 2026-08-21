from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from geofem_app.cli import run_solve
from geofem_app.fem2d import mesh_from_config
from geofem_app.output_location import resolve_analysis_output_dir
from geofem_app.samples import plane_strain_quad4_sample


class OutputLocationTests(unittest.TestCase):
    def test_resolver_supports_same_as_input_custom_and_legacy_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "入力" / "case.yaml"
            input_path.parent.mkdir()
            self.assertEqual(
                resolve_analysis_output_dir(input_path, {"root_policy": "same_as_input", "run_folder": "{input_stem}_run_fixed"}, root),
                input_path.parent / "case_run_fixed",
            )
            self.assertEqual(
                resolve_analysis_output_dir(input_path, {"root_policy": "custom", "directory": "out", "run_folder": "r1"}, root),
                input_path.parent / "out" / "r1",
            )
            self.assertEqual(
                resolve_analysis_output_dir(input_path, {"directory": "legacy_results"}, root),
                input_path.parent / "legacy_results",
            )
            explicit = root / "explicit"
            self.assertEqual(resolve_analysis_output_dir(input_path, {"root_policy": "same_as_input"}, root, explicit_out=explicit), explicit)

    def test_cli_same_as_input_writes_manifest_and_respects_vtk_format_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "日本語入力"
            input_dir.mkdir()
            cfg = plane_strain_quad4_sample()
            cfg["output"] = {
                "root_policy": "same_as_input",
                "run_folder": "{input_stem}_run_fixed",
                "formats": ["csv"],
                "write_log": True,
                "lazy_reports": True,
            }
            input_path = input_dir / "case.yaml"
            input_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

            self.assertEqual(run_solve(input_path), 0)
            out = input_dir / "case_run_fixed"
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "run.log").exists())
            self.assertTrue((out / "run_manifest.json").exists())
            self.assertTrue(list(out.rglob("*.csv")))
            self.assertFalse(list(out.rglob("*.vtk")))

    def test_cli_legacy_output_directory_remains_exact_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            cfg = plane_strain_quad4_sample()
            cfg["output"] = {"directory": "legacy_results", "formats": ["csv"], "lazy_reports": True}
            input_path = input_dir / "case.yaml"
            input_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

            self.assertEqual(run_solve(input_path), 0)
            self.assertTrue((input_dir / "legacy_results" / "summary.json").exists())
            self.assertTrue((input_dir / "legacy_results" / "run_manifest.json").exists())

    def test_external_gmsh_mesh_is_resolved_relative_to_input_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh_dir = root / "mesh"
            mesh_dir.mkdir()
            gmsh = mesh_dir / "quad.msh"
            gmsh.write_text(
                "\n".join(
                    [
                        "$MeshFormat",
                        "2.2 0 8",
                        "$EndMeshFormat",
                        "$Nodes",
                        "4",
                        "1 0 0 0",
                        "2 1 0 0",
                        "3 1 1 0",
                        "4 0 1 0",
                        "$EndNodes",
                        "$Elements",
                        "1",
                        "1 3 2 1 1 1 2 3 4",
                        "$EndElements",
                    ]
                ),
                encoding="utf-8",
            )
            mesh = mesh_from_config({"mesh": {"source": "external", "path": "mesh/quad.msh", "format": "gmsh", "base_dir": str(root)}})
            self.assertEqual(mesh.node_ids, ["1", "2", "3", "4"])
            self.assertEqual(len(mesh.elements), 1)
            self.assertEqual(mesh.elements[0].type, "QUAD4")


if __name__ == "__main__":
    unittest.main()
