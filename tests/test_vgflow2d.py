from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from geofem_app.cli import run_solve
from geofem_app.fem2d import mesh_from_config
from geofem_app.mesh_coupling import (
    apply_scalar_projection_plan,
    build_scalar_projection_plan,
    build_quad8_pressure_load_report,
    diagnose_quad8_to_quad4_downgrade,
    downgrade_quad8_mesh_to_quad4,
    interpolate_quad4_nodal_values_to_quad8,
    project_scalar_values_between_meshes,
    upgrade_quad4_mesh_to_quad8,
    write_mesh_coupling_manifest,
    write_mesh_projection_manifest,
    write_quad8_pressure_load_report,
)
from geofem_app.mesh_coupling_workflow import (
    build_coupled_post_comparison,
    build_geofeas_stage_handoff,
    build_material_layer_dictionary,
    build_minimum_mesh_coupling_benchmark,
    mesh_coupling_api_contract,
    write_coupled_post_comparison,
    write_geofeas_stage_handoff,
    write_material_layer_dictionary,
    write_mesh_coupling_api_contract,
    write_mesh_coupling_benchmark,
)
from geofem_app.vgflow2d import (
    _assemble_vgflow_matrices_python,
    _assemble_vgflow_matrices_quad4_numba,
    _assemble_vgflow_matrices_quad8_numba,
    _assemble_vgflow_matrices_tri3_numba,
    _assemble_vgflow_matrices_tri6_numba,
    read_vgflow_curve_file,
    read_vgflow_project_package,
    solve_vgflow2d_config,
    vgflow_design_template_catalog,
    vgflow_mesh_template_catalog,
    vgflow_materials_from_config,
    vgflow_pre_template_catalog,
    vgflow_unsaturated_public_catalog,
    write_vgflow_curve_file,
)


class VGFlow2DPublicSubstituteTests(unittest.TestCase):
    def test_quad4_vgflow_numba_assembly_matches_python_reference(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["materials"]["soil"]["seepage"].update(
            {
                "kx": 2.0e-5,
                "ky": 5.0e-6,
                "angle_deg": 18.0,
                "specific_storage": 0.02,
                "unsaturated": {
                    "model": "van_genuchten",
                    "alpha": 3.0,
                    "n": 1.6,
                    "theta_r": 0.08,
                    "theta_s": 0.43,
                },
            }
        )
        mesh = mesh_from_config(cfg)
        materials = vgflow_materials_from_config(cfg)
        head = np.linspace(0.4, 1.3, len(mesh.node_ids))

        fast = _assemble_vgflow_matrices_quad4_numba(mesh, materials, head, "vertical")
        reference = _assemble_vgflow_matrices_python(mesh, materials, head, "vertical")

        self.assertIsNotNone(fast)
        self.assertTrue(np.allclose(fast[0].toarray(), reference[0].toarray(), rtol=1.0e-12, atol=1.0e-18))
        self.assertTrue(np.allclose(fast[1].toarray(), reference[1].toarray(), rtol=1.0e-12, atol=1.0e-18))

    def test_quad4_vgflow_table_material_numba_assembly_matches_python_reference(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["materials"]["soil"]["seepage"].update(
            {
                "kx": 2.5e-5,
                "ky": 6.0e-6,
                "angle_deg": -12.0,
                "specific_storage": 0.015,
                "unsaturated": {
                    "model": "table",
                    "table": [
                        {"pressure_head": -2.0, "theta": 0.20, "kr": 0.05},
                        {"pressure_head": -0.5, "theta": 0.35, "kr": 0.40},
                        {"pressure_head": 0.0, "theta": 0.45, "kr": 1.00},
                    ],
                },
            }
        )
        mesh = mesh_from_config(cfg)
        materials = vgflow_materials_from_config(cfg)
        head = np.linspace(-0.6, 1.6, len(mesh.node_ids))

        fast = _assemble_vgflow_matrices_quad4_numba(mesh, materials, head, "vertical")
        reference = _assemble_vgflow_matrices_python(mesh, materials, head, "vertical")

        self.assertIsNotNone(fast)
        self.assertTrue(np.allclose(fast[0].toarray(), reference[0].toarray(), rtol=1.0e-12, atol=1.0e-18))
        self.assertTrue(np.allclose(fast[1].toarray(), reference[1].toarray(), rtol=1.0e-12, atol=1.0e-18))

    def test_quad8_vgflow_numba_assembly_matches_python_reference(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["element_type"] = "QUAD8"
        cfg["mesh"]["nx"] = 2
        cfg["materials"]["soil"]["seepage"].update(
            {
                "kx": 1.7e-5,
                "ky": 8.0e-6,
                "angle_deg": 9.0,
                "specific_storage": 0.01,
                "unsaturated": {
                    "model": "table",
                    "table": [
                        {"pressure_head": -3.0, "theta": 0.18, "kr": 0.03},
                        {"pressure_head": -1.0, "theta": 0.31, "kr": 0.25},
                        {"pressure_head": 0.0, "theta": 0.46, "kr": 1.00},
                    ],
                },
            }
        )
        mesh = mesh_from_config(cfg)
        materials = vgflow_materials_from_config(cfg)
        head = np.linspace(-0.2, 1.8, len(mesh.node_ids))

        fast = _assemble_vgflow_matrices_quad8_numba(mesh, materials, head, "vertical")
        reference = _assemble_vgflow_matrices_python(mesh, materials, head, "vertical")

        self.assertIsNotNone(fast)
        self.assertTrue(np.allclose(fast[0].toarray(), reference[0].toarray(), rtol=1.0e-12, atol=1.0e-18))
        self.assertTrue(np.allclose(fast[1].toarray(), reference[1].toarray(), rtol=1.0e-12, atol=1.0e-18))

    def test_tri_vgflow_numba_assembly_matches_python_reference(self) -> None:
        for element_type, kernel in (("TRI3", _assemble_vgflow_matrices_tri3_numba), ("TRI6", _assemble_vgflow_matrices_tri6_numba)):
            with self.subTest(element_type=element_type):
                cfg = _vgflow_base_config()
                cfg["mesh"]["element_type"] = element_type
                cfg["mesh"]["nx"] = 2
                cfg["materials"]["soil"]["seepage"].update(
                    {
                        "kx": 2.1e-5,
                        "ky": 9.0e-6,
                        "angle_deg": 6.0,
                        "specific_storage": 0.012,
                        "unsaturated": {
                            "model": "van_genuchten",
                            "alpha": 2.5,
                            "n": 1.8,
                            "theta_r": 0.07,
                            "theta_s": 0.44,
                        },
                    }
                )
                mesh = mesh_from_config(cfg)
                materials = vgflow_materials_from_config(cfg)
                head = np.linspace(0.1, 1.4, len(mesh.node_ids))

                fast = kernel(mesh, materials, head, "vertical")
                reference = _assemble_vgflow_matrices_python(mesh, materials, head, "vertical")

                self.assertIsNotNone(fast)
                self.assertTrue(np.allclose(fast[0].toarray(), reference[0].toarray(), rtol=1.0e-12, atol=1.0e-18))
                self.assertTrue(np.allclose(fast[1].toarray(), reference[1].toarray(), rtol=1.0e-12, atol=1.0e-18))

    def test_van_genuchten_material_and_steady_head_outputs_prs_ptn(self) -> None:
        cfg = _vgflow_base_config()
        cfg["vgflow2d"] = {
            "mode": "steady",
            "problem_type": "vertical",
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            self.assertEqual(len(result.steps), 1)
            self.assertTrue((out / "vgflow_node_results.csv").exists())
            self.assertTrue((out / "vgflow_element_results.csv").exists())
            self.assertTrue((out / "vgflow_waterline.PRS").exists())
            self.assertTrue((out / "vgflow_potential.PTN").exists())
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("richards_picard_total_head", summary["features"])
            with (out / "vgflow_element_results.csv").open(encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertGreater(float(row["velocity_x_m_s"]), 0.0)
            self.assertAlmostEqual(float(row["hydraulic_gradient_x"]), -1.0, places=8)

    def test_transient_rainfall_seepage_face_and_unsaturated_state_are_reported(self) -> None:
        cfg = _vgflow_base_config()
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "dt": 0.25,
            "steps": 2,
            "known_head_bcs": [{"set": "bottom", "head": 0.2}],
            "rainfall": {"set": "top", "rainfall": 3.6, "unit": "mm/hr"},
            "seepage_faces": [{"set": "top", "pressure_head": 0.0}],
            "initial_wetting_surface": [[0.0, 0.2], [1.0, 0.2]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_vgflow2d_config(cfg, tmp)
            self.assertEqual(len(result.steps), 2)
            self.assertTrue(all(step.iteration_count >= 1 for step in result.steps))
            with (Path(tmp) / "vgflow_node_results.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            top_rows = [row for row in rows if row["node_id"] in {"3", "4"}]
            self.assertTrue(any(float(row["saturation"]) < 1.0 for row in top_rows))
            self.assertTrue(any(float(row["pressure_head_m"]) < 0.0 for row in top_rows))

    def test_table_unsaturated_material_and_cli_route(self) -> None:
        cfg = _vgflow_base_config()
        cfg["materials"]["soil"]["seepage"]["unsaturated"] = {
            "model": "table",
            "table": [
                {"pressure_head": -2.0, "theta": 0.20, "kr": 0.05},
                {"pressure_head": 0.0, "theta": 0.45, "kr": 1.0},
            ],
        }
        materials = vgflow_materials_from_config(cfg)
        self.assertEqual(materials["soil"].unsaturated_model, "table")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "vgflow.yaml"
            input_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            out = root / "out"
            self.assertEqual(run_solve(input_path, output_dir=str(out)), 0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], "geofem.vgflow2d.public_substitute.v1")
            self.assertTrue((out / "case_manifest.json").exists())

    def test_curve_file_drives_boundary_time_series_and_exports_catalog(self) -> None:
        cfg = _vgflow_base_config()
        cfg["analysis"]["mode"] = "transient"
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0],
            "known_head_bcs": [{"set": "left", "head_curve_file": ""}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curve_path = root / "left_head.fcd"
            curve_path.write_text("hr head\n0 2.0\n1 3.0\n", encoding="utf-8")
            cfg["vgflow2d"]["known_head_bcs"][0]["head_curve_file"] = str(curve_path)
            result = solve_vgflow2d_config(cfg, root / "out")
            left_nodes = [result.mesh.node_index[nid] for nid in result.mesh.node_sets["left"]]
            self.assertEqual(len(result.steps), 2)
            self.assertTrue(all(np.isclose(result.steps[-1].total_head[left_nodes], 3.0)))
            boundary_curves = root / "out" / "vgflow_boundary_curves.csv"
            self.assertTrue(boundary_curves.exists())
            with boundary_curves.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["value_key"], "head")
            self.assertEqual(rows[-1]["source"], str(curve_path))
            for name in ("vgflow_curve_package.json", "vgflow_curve_package.csv", "vgflow_curve_package.html"):
                self.assertTrue((root / "out" / name).exists(), name)
            curve_package = json.loads((root / "out" / "vgflow_curve_package.json").read_text(encoding="utf-8"))
            self.assertFalse(curve_package["commercial_curve_binary_equivalence"])
            self.assertIn("open_curve_manifest", curve_package["features"])
            self.assertEqual(curve_package["curves"][0]["value_key"], "head")
            self.assertEqual(curve_package["curves"][0]["point_count"], 2)
            self.assertEqual(curve_package["curves"][0]["source_kind"], "file")
            summary = json.loads((root / "out" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("open_boundary_curve_package_manifest", summary["features"])
            self.assertIn("curve_package_json", summary["artifacts"])

    def test_exchange_package_selects_transient_waterline_and_potential_times(self) -> None:
        cfg = _vgflow_base_config()
        cfg["analysis"]["mode"] = "transient"
        cfg["mesh"]["nx"] = 2
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0, 2.0],
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "exchange": {"selected_times": [1.0], "targets": ["geofeas", "slope_stability"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_exchange_manifest.json",
                "vgflow_exchange_time_catalog.csv",
                "vgflow_exchange_operation_log.json",
                "vgflow_exchange_time_selection.html",
                "vgflow_geofeas_selected_waterline.PRS",
                "vgflow_geofeas_selected_potential.PTN",
                "vgflow_slope_selected_waterlines.csv",
                "vgflow_slope_selected_potentials.csv",
            ):
                self.assertTrue((out / name).exists(), name)
            manifest = json.loads((out / "vgflow_exchange_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"]["selected_steps"], [1])
            self.assertIn("selected_geofeas_waterline_prs", manifest["features"])
            self.assertIn("shared_hydro_exchange_engine", manifest["features"])
            self.assertEqual(manifest["shared_engine"]["module"], "geofem_app.hydro_exchange")
            with (out / "vgflow_exchange_time_catalog.csv").open(encoding="utf-8", newline="") as f:
                catalog = list(csv.DictReader(f))
            self.assertEqual([row["selected"] for row in catalog], ["False", "True", "False"])
            with (out / "vgflow_geofeas_selected_waterline.PRS").open(encoding="utf-8", newline="") as f:
                prs_rows = [row for row in csv.reader(f) if row and not row[0].startswith("#")]
            self.assertEqual(prs_rows[1][0], "1")
            with (out / "vgflow_slope_selected_waterlines.csv").open(encoding="utf-8", newline="") as f:
                slope_rows = list(csv.DictReader(f))
            self.assertTrue(slope_rows)
            self.assertTrue(all(row["purpose"] == "slope_stability_phreatic_line" for row in slope_rows))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("transient_exchange_time_selection_package", summary["features"])
            self.assertIn("shared_hydro_exchange_engine", summary["features"])
            self.assertEqual(summary["shared_engines"]["hydro_exchange"]["schema"], "geofem.shared_hydro_exchange.public_substitute.v1")
            self.assertIn("exchange_manifest", summary["artifacts"])
            report_manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("exchange_manifest", report_manifest["source_artifacts"])

    def test_quad4_quad8_mesh_coupling_upgrade_downgrade_manifest_and_values(self) -> None:
        cfg = _vgflow_base_config()
        quad4 = mesh_from_config(cfg)
        quad8, manifest = upgrade_quad4_mesh_to_quad8(quad4)

        self.assertEqual(manifest["schema"], "geofem.mesh_coupling.quad4_quad8.v1")
        self.assertEqual(manifest["mode"], "quad4_to_quad8")
        self.assertTrue(manifest["diagnostics"]["area_match"])
        self.assertEqual(quad8.elements[0].type, "QUAD8")
        self.assertEqual(quad8.elements[0].nodes, ("1", "2", "4", "3", "m_1_2", "m_2_4", "m_3_4", "m_1_3"))
        self.assertEqual(quad8.node_sets["bottom"], ["1", "m_1_2", "2"])

        mapped = interpolate_quad4_nodal_values_to_quad8(quad4, quad8, {"1": 10.0, "2": 20.0, "3": 30.0, "4": 40.0})
        self.assertAlmostEqual(mapped["1"], 10.0)
        self.assertAlmostEqual(mapped["m_1_2"], 15.0)
        self.assertAlmostEqual(mapped["m_2_4"], 30.0)
        self.assertAlmostEqual(mapped["m_3_4"], 35.0)
        self.assertAlmostEqual(mapped["m_1_3"], 20.0)

        downgraded, down_manifest = downgrade_quad8_mesh_to_quad4(quad8)
        self.assertEqual(down_manifest["mode"], "quad8_to_quad4")
        self.assertTrue(down_manifest["diagnostics"]["area_match"])
        self.assertEqual(downgraded.elements[0].type, "QUAD4")
        self.assertEqual(downgraded.elements[0].nodes, ("1", "2", "4", "3"))
        self.assertEqual(downgraded.node_sets["bottom"], ["1", "2"])

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_mesh_coupling_manifest(tmp, manifest)
            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertIn("midside_linear_interpolation", payload["features"])

    def test_quad8_vgflow_accepts_three_node_hydraulic_boundary_edges(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["element_type"] = "QUAD8"
        cfg["vgflow2d"] = {
            "mode": "steady",
            "problem_type": "vertical",
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "flux_bcs": [{"edge": ["1", "5", "2"], "flux": 1.0e-6}],
            "initial_water_level": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_vgflow2d_config(cfg, tmp)
            self.assertEqual(result.mesh.node_sets["bottom"], ["1", "5", "2"])
            self.assertEqual(result.mesh.elements[0].type, "QUAD8")
            with (Path(tmp) / "vgflow_node_results.csv").open(encoding="utf-8", newline="") as f:
                node_ids = {row["node_id"] for row in csv.DictReader(f)}
            self.assertIn("5", node_ids)

    def test_nonmatching_mesh_projection_reports_shape_and_fallback_diagnostics(self) -> None:
        source_cfg = _vgflow_base_config()
        source = mesh_from_config(source_cfg)
        source_values = {
            nid: float(source.coords[source.node_index[nid], 0] + 2.0 * source.coords[source.node_index[nid], 1])
            for nid in source.node_ids
        }
        target_cfg = _vgflow_base_config()
        target_cfg["mesh"] = dict(target_cfg["mesh"], nx=2, ny=2, element_type="QUAD8")
        target = mesh_from_config(target_cfg)

        projected, manifest = project_scalar_values_between_meshes(source, target, source_values, locations="both")
        self.assertEqual(manifest["schema"], "geofem.mesh_coupling.projection.v1")
        self.assertIn("nonmatching_mesh_scalar_projection", manifest["features"])
        self.assertIn("bbox_prefiltered_source_element_search", manifest["features"])
        self.assertIn("reusable_projection_weight_cache", manifest["features"])
        self.assertIn("parallel_projection_weight_application", manifest["features"])
        self.assertEqual(manifest["diagnostics"]["fallback_count"], 0)
        self.assertGreater(len([row for row in manifest["projection_map"] if row["location_type"] == "integration_point"]), 0)
        for nid in target.node_ids:
            xy = target.coords[target.node_index[nid]]
            self.assertAlmostEqual(projected[nid], float(xy[0] + 2.0 * xy[1]), places=9)

        plan = build_scalar_projection_plan(source, target, locations="both")
        self.assertEqual(plan.source_indices.shape[0], manifest["diagnostics"]["location_count"])
        projected_again, plan_manifest = apply_scalar_projection_plan(plan, source_values)
        self.assertEqual(projected_again, projected)
        doubled_values = {nid: value * 2.0 for nid, value in source_values.items()}
        doubled_projected, _doubled_manifest = apply_scalar_projection_plan(plan, doubled_values)
        for nid in target.node_ids:
            xy = target.coords[target.node_index[nid]]
            self.assertAlmostEqual(doubled_projected[nid], 2.0 * float(xy[0] + 2.0 * xy[1]), places=9)
        self.assertEqual(plan_manifest["diagnostics"]["fallback_count"], 0)

        outside_cfg = _vgflow_base_config()
        outside_cfg["mesh"] = dict(outside_cfg["mesh"], x_range=[0.0, 1.2], nx=2, ny=1, element_type="QUAD4")
        outside_target = mesh_from_config(outside_cfg)
        _outside_values, outside_manifest = project_scalar_values_between_meshes(source, outside_target, source_values)
        self.assertGreater(outside_manifest["diagnostics"]["fallback_count"], 0)
        self.assertTrue(any(row["status"] == "nearest_fallback" for row in outside_manifest["projection_map"]))

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_mesh_projection_manifest(tmp, manifest)
            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["diagnostics"]["location_count"], manifest["diagnostics"]["location_count"])

    def test_quad8_to_quad4_downgrade_reports_curved_edge_loss(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [1.0, 1.0],
                    "4": [0.0, 1.0],
                    "5": [0.5, 0.0],
                    "6": [1.0, 0.5],
                    "7": [0.5, 1.2],
                    "8": [0.0, 0.5],
                },
                "elements": [{"id": "e1", "type": "QUAD8", "material": "soil", "nodes": ["1", "2", "3", "4", "5", "6", "7", "8"]}],
            },
        }
        mesh = mesh_from_config(cfg)
        diagnostics = diagnose_quad8_to_quad4_downgrade(mesh, tolerance=0.05)
        self.assertEqual(diagnostics["schema"], "geofem.mesh_coupling.quad8_downgrade_diagnostics.v1")
        self.assertEqual(diagnostics["diagnostics"]["boundary_edge_count"], 4)
        self.assertEqual(diagnostics["diagnostics"]["curved_edge_count"], 1)
        top_edge = next(row for row in diagnostics["edge_diagnostics"] if row["mid_node"] == "7")
        self.assertEqual(top_edge["status"], "warning")
        self.assertAlmostEqual(float(top_edge["sagitta"]), 0.2)

        _downgraded, manifest = downgrade_quad8_mesh_to_quad4(mesh)
        self.assertIn("quad8_to_quad4_curved_edge_diagnostics", manifest["features"])
        self.assertEqual(manifest["diagnostics"]["curved_edge_count"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_mesh_coupling_manifest(tmp, manifest)
            self.assertTrue(Path(paths["edge_diagnostics_csv"]).exists())

    def test_quad8_pressure_load_report_preserves_quadratic_edge_resultant(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["element_type"] = "QUAD8"
        mesh = mesh_from_config(cfg)
        pressures = {nid: 0.0 for nid in mesh.node_ids}
        for nid in mesh.node_sets["top"]:
            pressures[nid] = 10.0

        report = build_quad8_pressure_load_report(mesh, pressures)
        self.assertEqual(report["schema"], "geofem.mesh_coupling.quad8_pressure_load_report.v1")
        self.assertIn("resultant_conservation_check", report["features"])
        self.assertEqual(report["diagnostics"]["edge_count"], 4)
        self.assertAlmostEqual(report["diagnostics"]["max_conservation_error"], 0.0, places=12)
        top = next(row for row in report["pressure_loads"] if row["mid_node"] == "7")
        self.assertAlmostEqual(float(top["resultant_normal"]), 10.0, places=10)
        self.assertAlmostEqual(float(top["mid_force_y"]), 10.0 * 8.0 / 15.0 + 10.0 / 15.0 + 10.0 / 15.0, places=10)
        self.assertGreater(float(top["resultant_y"]), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_quad8_pressure_load_report(tmp, report)
            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)

    def test_vgflow_geofeas_coupling_workflow_stage_material_post_contract_and_benchmark(self) -> None:
        cfg = _vgflow_base_config()
        cfg["analysis"]["mode"] = "transient"
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0],
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "exchange": {"selected_steps": [1]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_geofeas_mesh_coupling.json",
                "vgflow_geofeas_coupling_ui_profile.json",
                "vgflow_geofeas_stage_handoff.json",
                "vgflow_geofeas_stage_handoff_hydro_bcs.csv",
                "vgflow_geofeas_material_layers.json",
                "vgflow_geofeas_coupling_benchmark.json",
                "vgflow_geofeas_coupling_api_contract.json",
            ):
                self.assertTrue((out / name).exists(), name)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("vgflow_geofeas_mesh_coupling_workflow", summary["features"])
            self.assertIn("stage_handoff_json", summary["artifacts"])

            geofeas_mesh, _manifest = upgrade_quad4_mesh_to_quad8(result.mesh)
            handoff = build_geofeas_stage_handoff(result.mesh, geofeas_mesh, result.steps, "vertical", selected_steps=[1], existing_stages=[{"name": "VGFlow_0001"}])
            self.assertEqual(handoff["diagnostics"]["updated_stage_count"], 1)
            self.assertEqual(handoff["stage_rows"][0]["pressure_bc_count"], len(geofeas_mesh.node_ids))
            dictionary = build_material_layer_dictionary(result.mesh, geofeas_mesh, result.materials, result.materials)
            self.assertEqual(dictionary["diagnostics"]["warning_count"], 0)

            source_head = {nid: float(result.steps[-1].total_head[result.mesh.node_index[nid]]) for nid in result.mesh.node_ids}
            target_head = interpolate_quad4_nodal_values_to_quad8(result.mesh, geofeas_mesh, source_head)
            comparison = build_coupled_post_comparison(result.mesh, geofeas_mesh, {"total_head": source_head}, {"total_head": target_head})
            self.assertEqual(comparison["diagnostics"]["field_pair_count"], 1)
            self.assertAlmostEqual(comparison["diagnostics"]["max_abs_difference"], 0.0, places=10)

            benchmark = build_minimum_mesh_coupling_benchmark()
            self.assertEqual(benchmark["diagnostics"]["warning_count"], 0)
            contract = mesh_coupling_api_contract()
            self.assertIn("geofem.mesh_coupling.api_contract.v1", contract["schema"])

            extra_paths = []
            extra_paths.extend(write_geofeas_stage_handoff(out / "extra", handoff).values())
            extra_paths.extend(write_material_layer_dictionary(out / "extra", dictionary).values())
            extra_paths.extend(write_coupled_post_comparison(out / "extra", comparison).values())
            extra_paths.extend(write_mesh_coupling_benchmark(out / "extra", benchmark).values())
            extra_paths.extend(write_mesh_coupling_api_contract(out / "extra", contract).values())
            for path in extra_paths:
                self.assertTrue(Path(path).exists(), path)

    def test_boundary_diagnostics_report_rainfall_excess_overlap_and_curve_alignment(self) -> None:
        cfg = _vgflow_base_config()
        cfg["analysis"]["mode"] = "transient"
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0, 2.0],
            "known_head_bcs": [{"set": "left", "head_curve": [{"time": 0.0, "head": 2.0}, {"time": 1.0, "head": 2.0}]}],
            "rainfall": {"set": "left", "rainfall": 3600.0, "unit": "mm/hr"},
            "seepage_faces": [{"set": "left", "pressure_head": 0.0}],
            "initial_water_level": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_boundary_diagnostics.json",
                "vgflow_boundary_diagnostics.csv",
                "vgflow_boundary_diagnostics.html",
            ):
                self.assertTrue((out / name).exists(), name)
            payload = json.loads((out / "vgflow_boundary_diagnostics.json").read_text(encoding="utf-8"))
            checks = {row["check"] for row in payload["diagnostics"]}
            self.assertIn("rainfall_infiltration_capacity", checks)
            self.assertIn("fixed_head_flux_overlap", checks)
            self.assertIn("rainfall_seepage_face_overlap", checks)
            self.assertIn("boundary_curve_time_range", checks)
            self.assertTrue(any(row["status"] == "warning" for row in payload["diagnostics"]))
            rainfall_rows = [row for row in payload["diagnostics"] if row["check"] == "rainfall_infiltration_capacity"]
            self.assertTrue(any(float(row["excess"]) > 0.0 for row in rainfall_rows))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("boundary_rainfall_runoff_curve_alignment_diagnostics", summary["features"])
            self.assertIn("boundary_diagnostics_json", summary["artifacts"])
            report_manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("boundary_diagnostics_json", report_manifest["source_artifacts"])

    def test_cad_import_diagnostics_scale_dxf_and_calibrate_raster_lines(self) -> None:
        cfg = _vgflow_base_config()
        cfg["vgflow2d"] = {
            "mode": "steady",
            "problem_type": "vertical",
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxf_path = root / "profile.dxf"
            dxf_path.write_text(
                "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\nground\n10\n0\n20\n0\n11\n1000\n21\n500\n0\nENDSEC\n0\nEOF\n",
                encoding="utf-8",
            )
            cfg["vgflow2d"]["cad_import"] = {
                "files": [{"path": str(dxf_path), "source_unit": "mm", "target_unit": "m"}],
                "raster_images": [
                    {
                        "source": "scan.png",
                        "calibration": [
                            {"pixel": [0.0, 0.0], "world": [0.0, 0.0]},
                            {"pixel": [1000.0, 0.0], "world": [10.0, 0.0]},
                        ],
                        "traced_polylines": [{"name": "stratum", "points": [[0.0, 0.0], [500.0, 100.0], [1000.0, 0.0]]}],
                    }
                ],
            }
            solve_vgflow2d_config(cfg, root / "out")
            out = root / "out"
            for name in (
                "vgflow_cad_import_diagnostics.json",
                "vgflow_cad_import_diagnostics.csv",
                "vgflow_cad_import_model_lines.csv",
                "vgflow_cad_import.html",
            ):
                self.assertTrue((out / name).exists(), name)
            payload = json.loads((out / "vgflow_cad_import_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["shared_engine"]["module"], "geofem_app.cad_import")
            self.assertEqual(payload["shared_engine"]["document_entrypoint"], "parse_cad_file_document")
            checks = {row["check"] for row in payload["diagnostics"]}
            self.assertIn("shared_geofem_cad_import_engine", checks)
            self.assertIn("cad_attribute_summary", checks)
            self.assertIn("dxf_mm_to_m_scale_correction", checks)
            self.assertIn("raster_image_calibration", checks)
            self.assertIn("raster_traced_polyline_to_model_lines", checks)
            with (out / "vgflow_cad_import_model_lines.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            cad_row = next(row for row in rows if row["source_type"] == "cad")
            self.assertAlmostEqual(float(cad_row["x2"]), 1.0)
            self.assertAlmostEqual(float(cad_row["y2"]), 0.5)
            raster_row = next(row for row in rows if row["source_type"] == "raster")
            self.assertAlmostEqual(float(raster_row["x2"]), 5.0)
            self.assertAlmostEqual(float(raster_row["y2"]), 1.0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("shared_geofem_cad_import_engine", summary["features"])
            self.assertIn("cad_raster_import_scale_diagnostics", summary["features"])
            self.assertIn("cad_import_diagnostics_json", summary["artifacts"])
            report_manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("cad_import_diagnostics_json", report_manifest["source_artifacts"])

    def test_raster_auto_dark_line_extraction_builds_model_lines(self) -> None:
        cfg = _vgflow_base_config()
        cfg["vgflow2d"] = {
            "mode": "steady",
            "problem_type": "vertical",
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pgm_path = root / "scan.pgm"
            pixels = []
            for y in range(6):
                row = []
                for x in range(11):
                    row.append("0" if y == 2 else "255")
                pixels.append(" ".join(row))
            pgm_path.write_text("P2\n11 6\n255\n" + "\n".join(pixels) + "\n", encoding="ascii")
            cfg["vgflow2d"]["cad_import"] = {
                "raster_images": [
                    {
                        "source": str(pgm_path),
                        "auto_extract": True,
                        "threshold": 32,
                        "max_auto_points": 11,
                        "calibration": [
                            {"pixel": [0.0, 0.0], "world": [0.0, 0.0]},
                            {"pixel": [10.0, 0.0], "world": [10.0, 0.0]},
                        ],
                    }
                ],
            }
            solve_vgflow2d_config(cfg, root / "out")
            out = root / "out"
            payload = json.loads((out / "vgflow_cad_import_diagnostics.json").read_text(encoding="utf-8"))
            auto_rows = [row for row in payload["diagnostics"] if row["check"] == "raster_auto_dark_line_extraction"]
            self.assertEqual(auto_rows[0]["status"], "pass")
            self.assertEqual(auto_rows[0]["details"]["point_count"], 11)
            with (out / "vgflow_cad_import_model_lines.csv").open(encoding="utf-8", newline="") as f:
                rows = [row for row in csv.DictReader(f) if row["source_type"] == "raster_auto"]
            self.assertEqual(len(rows), 10)
            self.assertAlmostEqual(float(rows[0]["x1"]), 0.0)
            self.assertAlmostEqual(float(rows[0]["y1"]), 2.0)
            self.assertAlmostEqual(float(rows[-1]["x2"]), 10.0)
            self.assertAlmostEqual(float(rows[-1]["y2"]), 2.0)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("cad_raster_auto_strata_extraction", summary["features"])

    def test_open_vg2_surrogate_project_package_roundtrips_public_state(self) -> None:
        cfg = _vgflow_base_config()
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0],
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_public_project.VG2",
                "vgflow_public_project_manifest.json",
                "vgflow_public_project_inventory.csv",
                "vgflow_public_project.html",
            ):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual((out / "vgflow_public_project.VG2").read_bytes()[:2], b"PK")
            package = read_vgflow_project_package(out / "vgflow_public_project.VG2")
            self.assertEqual(package["manifest"]["schema"], "geofem.vgflow2d.project_package.public_substitute.v1")
            self.assertFalse(package["manifest"]["commercial_vg2_binary_equivalence"])
            self.assertEqual(package["manifest"]["step_count"], 2)
            self.assertEqual(len(package["model"]["nodes"]), 4)
            self.assertEqual(package["seepage"]["mode"], "transient")
            self.assertEqual(len(package["steps"]), 2)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("open_vg2_surrogate_project_package", summary["features"])
            self.assertIn("project_package_vg2", summary["artifacts"])

    def test_public_unsaturated_catalog_preset_and_curve_roundtrip(self) -> None:
        catalog = vgflow_unsaturated_public_catalog()
        self.assertIn("dam_core_low_permeability", {row["id"] for row in catalog})
        cfg = _vgflow_base_config()
        cfg["materials"]["soil"]["seepage"].pop("unsaturated")
        cfg["materials"]["soil"]["seepage"]["unsaturated_preset"] = "dam_core"
        material = vgflow_materials_from_config(cfg)["soil"]
        self.assertEqual(material.unsaturated_model, "van_genuchten")
        self.assertAlmostEqual(material.alpha, 1.5)
        self.assertAlmostEqual(material.n, 3.0)
        self.assertAlmostEqual(material.theta_r, 0.30)
        self.assertAlmostEqual(material.theta_s, 0.70)
        with tempfile.TemporaryDirectory() as tmp:
            curve_path = Path(tmp) / "rainfall.qcd"
            write_vgflow_curve_file(
                [{"time": 0.0, "rainfall": 0.0}, {"time": 1.0, "rainfall": 3.6}],
                curve_path,
                value_field="rainfall",
            )
            rows = read_vgflow_curve_file(curve_path, value_field="rainfall")
        self.assertEqual(rows[1]["rainfall"], 3.6)

    def test_post_contours_vectors_flowlines_sections_and_histories_are_written(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["mesh"]["ny"] = 2
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0],
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "post": {
                "contour_level_count": 3,
                "flowline_seed_count": 3,
                "flowline_max_points": 8,
                "flow_sections": [{"name": "center", "x": 0.5, "y_range": [0.0, 1.0]}],
                "history_nodes": ["1", "9"],
                "history_elements": ["1"],
                "thickness_m": 2.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("post_contours_vectors_flowlines_sections", summary["features"])
            for name in (
                "vgflow_post_nodal_fields.csv",
                "vgflow_post_contours.csv",
                "vgflow_flow_vectors.csv",
                "vgflow_flowlines.csv",
                "vgflow_section_flows.csv",
                "vgflow_time_history.csv",
                "vgflow_post_table_schema.json",
                "vgflow_post_node_table.tsv",
                "vgflow_post_element_table.tsv",
                "vgflow_post_section_flow_units.csv",
                "vgflow_post_animation_manifest.json",
                "vgflow_post_animation_frames.csv",
                "vgflow_post_animation.html",
                "vgflow_post_animation.avi",
                "vgflow_post_animation_avi_manifest.json",
                "vgflow_post_manifest.json",
            ):
                self.assertTrue((out / name).exists(), name)
            with (out / "vgflow_post_contours.csv").open(encoding="utf-8", newline="") as f:
                contour_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["variable"] == "total_head_m" for row in contour_rows))
            with (out / "vgflow_flow_vectors.csv").open(encoding="utf-8", newline="") as f:
                vector_rows = list(csv.DictReader(f))
            self.assertTrue(any(float(row["velocity_abs_m_s"]) > 0.0 for row in vector_rows))
            with (out / "vgflow_section_flows.csv").open(encoding="utf-8", newline="") as f:
                flow_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["section"] == "center" and float(row["abs_flow_rate_m3_s_per_m"]) > 0.0 for row in flow_rows))
            with (out / "vgflow_time_history.csv").open(encoding="utf-8", newline="") as f:
                history_kinds = {row["kind"] for row in csv.DictReader(f)}
            self.assertEqual(history_kinds, {"node", "element"})
            schema = json.loads((out / "vgflow_post_table_schema.json").read_text(encoding="utf-8"))
            self.assertIn("Pressure head (m)", [row["label"] for row in schema["columns"]["node"]])
            self.assertIn("tab-separated values", schema["copy_format"])
            tsv_header = (out / "vgflow_post_node_table.tsv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("Velocity (m/s)", tsv_header)
            with (out / "vgflow_post_section_flow_units.csv").open(encoding="utf-8", newline="") as f:
                unit_rows = list(csv.DictReader(f))
            self.assertTrue(any(row["positive_direction"] == "+X" and float(row["thickness_m"]) == 2.0 for row in unit_rows))
            animation = json.loads((out / "vgflow_post_animation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(animation["frame_count"], 6)
            self.assertIn("time_varying_contour_frames", animation["features"])
            self.assertIn("direct_avi_animation_export", animation["features"])
            self.assertEqual((out / "vgflow_post_animation.avi").read_bytes()[:12], b"RIFF" + (out / "vgflow_post_animation.avi").read_bytes()[4:8] + b"AVI ")
            avi_manifest = json.loads((out / "vgflow_post_animation_avi_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(avi_manifest["format"], "AVI")
            self.assertEqual(avi_manifest["frame_count"], 2)
            self.assertFalse(avi_manifest["commercial_renderer_equivalence"])
            post_manifest = json.loads((out / "vgflow_post_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("clipboard_friendly_tsv_tables", post_manifest["features"])
            self.assertIn("direct_avi_animation_export", post_manifest["features"])
            self.assertIn("post_animation_manifest", post_manifest["artifacts"])
            self.assertIn("post_tables_units_copy_animation_flow_sign", summary["features"])
            self.assertIn("post_direct_avi_animation_export", summary["features"])
            self.assertEqual(len(result.steps), 2)

    def test_report_bundle_writes_html_pdf_manifest_and_selection(self) -> None:
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["mesh"]["ny"] = 2
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0],
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "report": {
                "sections": ["model", "mesh", "analysis", "materials", "boundaries", "post_outputs", "time_history"],
                "node_ids": ["1", "9"],
                "element_ids": ["1"],
                "time_range": [0.0, 1.0],
                "print_profile": {
                    "paper_size": "A3",
                    "orientation": "landscape",
                    "margins_mm": [10, 12, 10, 12],
                    "post_apply": ["post_contours", "flow_vectors"],
                    "figure_style": {"contour_palette": "rainbow", "vector_scale": 2.0, "show_legend": True},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_report_data.json",
                "vgflow_report_sections.csv",
                "vgflow_report.html",
                "vgflow_report.pdf",
                "vgflow_report_public_ppf_profile.json",
                "vgflow_report_public_ppf_profile.csv",
                "vgflow_report_public_ppf_profile.html",
                "vgflow_report_manifest.json",
            ):
                self.assertTrue((out / name).exists(), name)
            manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("direct_pdf_report", manifest["features"])
            self.assertIn("public_ppf_print_profile_substitute", manifest["features"])
            self.assertEqual(manifest["selection"]["node_ids"], ["1", "9"])
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("report_manifest_html_pdf", summary["features"])
            self.assertIn("report_public_ppf_print_profile_substitute", summary["features"])
            self.assertIn("alternative_spec_acceptance_profile", summary["features"])
            self.assertIn("vgflow_report_manifest", summary["artifacts"])
            self.assertIn("alternative_spec_json", summary["artifacts"])
            acceptance = json.loads((out / "vgflow_alternative_spec_acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(acceptance["acceptance_level"], "open_public_substitute_not_commercial_identity")
            self.assertFalse(acceptance["commercial_equivalence_claim"])
            self.assertEqual(acceptance["remaining_unmet_count"], 0)
            self.assertEqual({row["status"] for row in acceptance["rows"]}, {"accepted_public_substitute"})
            self.assertTrue((out / "vgflow_alternative_spec_acceptance.csv").exists())
            self.assertTrue((out / "vgflow_alternative_spec_acceptance.html").exists())
            profile = json.loads((out / "vgflow_report_public_ppf_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["page"]["paper_size"], "A3")
            self.assertEqual(profile["page"]["orientation"], "landscape")
            self.assertEqual([row["artifact_key"] for row in profile["post_apply"]], ["post_contours", "flow_vectors"])
            self.assertEqual(profile["figure_style"]["contour_palette"], "rainbow")
            html = (out / "vgflow_report.html").read_text(encoding="utf-8")
            self.assertIn("VGFlow 2D 公開代替帳票", html)
            with (out / "vgflow_report_sections.csv").open(encoding="utf-8", newline="") as f:
                sections = {row["section"]: row for row in csv.DictReader(f)}
            self.assertEqual(sections["post_outputs"]["enabled"], "True")

    def test_design_checks_write_gradient_courant_and_template_guidance(self) -> None:
        self.assertIn("cutoff_wall", {row["id"] for row in vgflow_design_template_catalog()})
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["mesh"]["ny"] = 2
        cfg["vgflow2d"] = {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0],
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "design_checks": {
                "critical_gradient": 0.5,
                "warning_ratio": 0.75,
                "max_courant": 1.0e-6,
                "templates": ["cutoff_wall", "wellpoint"],
                "head_difference_pairs": [{"name": "lr", "a": {"node": "1"}, "b": {"node": "3"}}],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_design_checks.json",
                "vgflow_design_checks.csv",
                "vgflow_design_checks.html",
                "vgflow_design_templates.json",
            ):
                self.assertTrue((out / name).exists(), name)
            payload = json.loads((out / "vgflow_design_checks.json").read_text(encoding="utf-8"))
            checks = {row["check"] for row in payload["checks"]}
            self.assertIn("two_point_head_difference", checks)
            self.assertIn("local_piping_boiling", checks)
            self.assertIn("courant_number", checks)
            self.assertIn("purpose_template", checks)
            self.assertTrue(any(row["status"] == "fail" for row in payload["checks"]))
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("design_checks_piping_boiling_courant_templates", summary["features"])
            self.assertIn("design_checks_json", summary["artifacts"])
            report_manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("design_checks_json", report_manifest["source_artifacts"])

    def test_pre_operation_log_templates_mesh_mode_and_selection_state_are_written(self) -> None:
        self.assertIn("semi_auto", {row["id"] for row in vgflow_pre_template_catalog()})
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["mesh"]["ny"] = 2
        cfg["vgflow2d"] = {
            "mode": "steady",
            "problem_type": "vertical",
            "mesh_mode": "semi_auto",
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "pre": {
                "workflow": "river_embankment_public_pre",
                "grid": {"enabled": True, "origin": [0.0, 0.0], "spacing": 0.5},
                "snap": {"enabled": True, "mode": "grid", "grid_size": 0.5},
                "straight_lines": [{"name": "base", "points": [[0.0, 0.0], [1.0, 0.0]], "horizontal": True}],
                "node_corrections": [{"node_id": "1", "x": 0.0, "y": 0.0}],
                "block_selection": [{"name": "B1", "mode": "box", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "action": "auto_block"}],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_pre_operation_log.json",
                "vgflow_pre_operation_log.csv",
                "vgflow_pre_state.json",
                "vgflow_pre_templates.json",
                "vgflow_ui_profile.json",
                "vgflow_ui_profile.csv",
                "vgflow_ui_profile.html",
            ):
                self.assertTrue((out / name).exists(), name)
            state = json.loads((out / "vgflow_pre_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["mesh_mode"], "semi_auto")
            self.assertEqual(state["straight_line_count"], 1)
            self.assertEqual(state["block_operations"][0]["name"], "B1")
            log = json.loads((out / "vgflow_pre_operation_log.json").read_text(encoding="utf-8"))["operation_log"]
            commands = {row["command"] for row in log}
            self.assertIn("mesh_mode", commands)
            self.assertIn("block_selection", commands)
            self.assertIn("condition_reset_prompt", commands)
            ui_profile = json.loads((out / "vgflow_ui_profile.json").read_text(encoding="utf-8"))
            self.assertFalse(ui_profile["source_policy"]["commercial_pixel_equivalence"])
            toolbar_commands = {command["id"] for group in ui_profile["toolbar_groups"] for command in group["commands"]}
            self.assertIn("mesh_split", toolbar_commands)
            self.assertIn("run", toolbar_commands)
            context_ids = {menu["id"] for menu in ui_profile["context_menus"]}
            self.assertIn("boundary_selection", context_ids)
            enable_ids = {row["id"] for row in ui_profile["button_enablement"]}
            self.assertIn("copy_selected_values", enable_ids)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("pre_operation_log_templates", summary["features"])
            self.assertIn("ui_toolbar_context_modal_state_profile", summary["features"])
            self.assertIn("pre_operation_log_json", summary["artifacts"])
            self.assertIn("ui_profile_json", summary["artifacts"])
            report_manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("pre_operation_log_json", report_manifest["source_artifacts"])
            self.assertIn("ui_profile_json", report_manifest["source_artifacts"])

    def test_mesh_generation_plan_quality_and_hydraulic_refinement_are_written(self) -> None:
        self.assertIn("river_embankment_standard", {row["id"] for row in vgflow_mesh_template_catalog()})
        cfg = _vgflow_base_config()
        cfg["mesh"]["nx"] = 2
        cfg["mesh"]["ny"] = 1
        cfg["vgflow2d"] = {
            "mode": "steady",
            "problem_type": "vertical",
            "mesh_mode": "quadrilateral_only",
            "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
            "initial_water_level": 1.0,
            "mesh_generation": {
                "mesh_mode": "quadrilateral_only",
                "river_embankment_template": True,
                "line_divisions": [{"name": "crest", "division_count": 4}, {"name": "slope", "division_width": 0.25}],
                "gradient_refinement_threshold": 0.2,
                "quality": {"max_aspect_ratio": 10.0, "min_angle_deg": 20.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            solve_vgflow2d_config(cfg, tmp)
            out = Path(tmp)
            for name in (
                "vgflow_mesh_plan.json",
                "vgflow_mesh_plan.csv",
                "vgflow_mesh_quality.json",
                "vgflow_mesh_quality.csv",
                "vgflow_mesh_quality.html",
                "vgflow_mesh_templates.json",
            ):
                self.assertTrue((out / name).exists(), name)
            payload = json.loads((out / "vgflow_mesh_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["plan"]["mesh_mode"], "quadrilateral_only")
            self.assertEqual(len(payload["plan"]["line_divisions"]), 2)
            self.assertTrue(payload["plan"]["selected_templates"])
            self.assertTrue(payload["hydraulic_refinement_recommendations"])
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("mesh_generation_plan_quality_diagnostics", summary["features"])
            self.assertIn("mesh_plan_json", summary["artifacts"])
            report_manifest = json.loads((out / "vgflow_report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("mesh_plan_json", report_manifest["source_artifacts"])


def _vgflow_base_config() -> dict[str, object]:
    return {
        "analysis": {"dimension": "2D", "type": "vgflow2d", "mode": "steady"},
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "nx": 1,
            "ny": 1,
            "element_type": "QUAD4",
            "material": "soil",
        },
        "materials": {
            "soil": {
                "model": "elastic",
                "E": 1000.0,
                "nu": 0.3,
                "seepage": {
                    "kx": 1.0e-5,
                    "ky": 1.0e-5,
                    "specific_storage": 1.0e-4,
                    "unsaturated": {
                        "model": "van_genuchten",
                        "alpha": 1.5,
                        "n": 2.0,
                        "theta_r": 0.1,
                        "theta_s": 0.45,
                    },
                },
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
