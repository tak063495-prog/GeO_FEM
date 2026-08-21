from __future__ import annotations

import copy
import getpass
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

import numpy as np

from geofem_app.analytic_boolean import build_analytic_curve_boolean_graph, classify_graph_boolean_operations, intersect_curves, operation_loop_polygons, parse_boolean_expression, regions_containing_point
from geofem_app.cad_dwg_converter import dwg_converter_command
from geofem_app.cad_gf1_payload import best_effort_decode, gf1_payload_text_candidates
from geofem_app.cad_import import CadImportError, export_sxf_document, parse_cad_document, parse_cad_file, parse_cad_file_document, parse_cad_lines, parse_dxf_document, parse_gf1_document, parse_gf1_document_bytes, parse_gf1_lines, parse_gf1_lines_bytes, parse_sxf_document, parse_sxf_lines, split_sxf_step_records, validate_dwg_converter_link, validate_sxf_roundtrip
from geofem_app.mesh_generation import classify_element_regions, curve_segment_count, first_region_indices, generate_quad_dominant_mesh, generate_rectilinear_polygon_paving_mesh, improve_mesh_quality, inside_domain, near_point_pairs, node_ids_on_boundary, normalize_mesh_split_lines, normalize_rectilinear_region, point_to_segment_distance, polygon_has_crossing_edges, polygons_any_overlap, polygons_overlap, project_nodes_to_boundaries, recombine_triangles_to_quads


class InputTemplateTests(unittest.TestCase):
    def test_slope_srm_drucker_prager_template_is_ready(self) -> None:
        from geofem_app.fem2d_config import plane_strain_materials, validate_2d_core_scope
        from geofem_app.fem2d_mesh import mesh_from_config
        from geofem_app.fem2d_solver import solve_plane_strain_config
        from geofem_app.fem2d_solver_controls import srm_factors
        from geofem_app.gui.file_preview import read_mapping_file_guarded
        from geofem_app.gui.workflow_guidance import build_workflow_guidance
        from geofem_app.input_diagnostics import diagnose_input_config

        cfg, _preview = read_mapping_file_guarded(Path("examples/slope_srm_drucker_prager.yaml"))
        validate_2d_core_scope(cfg)
        mesh = mesh_from_config(cfg)
        material = plane_strain_materials(cfg)["colluvium_dp"]
        factors = srm_factors(cfg["stages"][0]["srm"])
        diagnostics = diagnose_input_config(cfg, locale="ja")
        guidance = build_workflow_guidance(cfg, locale="ja")

        self.assertEqual(len(mesh.node_ids), 429)
        self.assertEqual(len(mesh.elements), 384)
        self.assertIn("slope_body", mesh.element_sets)
        self.assertIn("slope_face", mesh.node_sets)
        self.assertEqual(material.model, "drucker_prager")
        self.assertGreater(material.cohesion, 0.0)
        self.assertEqual([stage["type"] for stage in cfg["stages"]], ["srm"])
        self.assertEqual(cfg["stages"][0]["srm"]["search_mode"], "adaptive_bracket")
        self.assertAlmostEqual(factors[0], 0.5)
        self.assertIn(1.0, factors)
        self.assertAlmostEqual(factors[-1], 1.6)
        self.assertEqual(diagnostics["error_count"], 0)
        self.assertEqual(diagnostics["warning_count"], 0)
        ready = {row["id"]: row["completed"] for row in guidance["steps"]}
        for step in ("analysis", "geometry", "mesh", "materials", "boundary_conditions", "loads", "stages", "model_check"):
            self.assertTrue(ready[step])
        with tempfile.TemporaryDirectory() as tmpdir:
            for integration in ("FULL", "B-bar", "SRI"):
                variant = copy.deepcopy(cfg)
                variant["mesh"]["integration"] = integration
                variant["stages"][0]["srm"]["factors"] = [0.5]
                result = solve_plane_strain_config(variant, Path(tmpdir) / integration.replace("-", "_"))
                srm = result.stages[-1].solver_info["srm"]
                self.assertAlmostEqual(srm["factor_of_safety"], 0.5)
                self.assertEqual(len(srm["trials"]), 1)


class CadBoundaryModuleTests(unittest.TestCase):
    def test_boundary_node_detection_batches_point_segment_distances(self) -> None:
        nodes = {"a": [0.0, 0.0], "b": [0.5, 0.0], "c": [1.0, 0.0], "d": [0.5, 0.2]}
        segments = [((0.0, 0.0), (1.0, 0.0))]
        self.assertEqual(node_ids_on_boundary(nodes, segments), ["a", "b", "c"])

    def test_region_and_polygon_geometry_use_batched_kernels(self) -> None:
        regions = [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)], [(3.0, 0.0), (5.0, 0.0), (5.0, 2.0), (3.0, 2.0)]]
        points = np.array([[0.5, 0.5], [3.5, 0.5], [6.0, 0.5]], dtype=float)
        self.assertEqual(first_region_indices(points, regions).tolist(), [0, 1, -1])

        nodes = {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0], "5": [3.0, 0.0], "6": [4.0, 0.0], "7": [4.0, 1.0], "8": [3.0, 1.0]}
        elements = [{"id": "e1", "nodes": ["1", "2", "3", "4"]}, {"id": "e2", "nodes": ["5", "6", "7", "8"]}]
        self.assertEqual(classify_element_regions(nodes, elements, regions), {"region_1": ["e1"], "region_2": ["e2"]})
        self.assertTrue(polygon_has_crossing_edges([(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]))
        self.assertTrue(polygons_overlap(regions[0], [(1.0, 1.0), (2.5, 1.0), (2.5, 2.5), (1.0, 2.5)]))
        far_regions = [
            [(10.0 + index * 3.0, 0.0), (11.0 + index * 3.0, 0.0), (11.0 + index * 3.0, 1.0), (10.0 + index * 3.0, 1.0)]
            for index in range(20)
        ]
        self.assertFalse(polygons_any_overlap(far_regions))
        self.assertTrue(polygons_any_overlap([*far_regions, regions[0], [(1.0, 1.0), (2.5, 1.0), (2.5, 2.5), (1.0, 2.5)]]))

    def test_boundary_projection_batches_nearest_search(self) -> None:
        nodes = {"a": [0.5, 0.03], "b": [1.05, 0.5], "c": [3.0, 3.0]}
        projected, info = project_nodes_to_boundaries(
            nodes,
            (0.0, 2.0, 0.0, 2.0),
            [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
            [(1.5, 0.5, 0.5)],
            0.2,
            snap_tolerance=0.1,
        )
        self.assertGreaterEqual(info["projected_node_count"], 2)
        self.assertTrue(np.allclose(projected["a"], [0.5, 0.0]))
        self.assertLess(abs(math.hypot(projected["b"][0] - 1.5, projected["b"][1] - 0.5) - 0.5), 1.0e-12)
        self.assertEqual(projected["c"], [3.0, 3.0])

    def test_near_point_pairs_uses_sorted_grid_candidates(self) -> None:
        far = [(f"far_{index}", 10.0 + index * 4.0, 5.0) for index in range(40)]
        points = [("a", 0.0, 0.0), ("b", 5.0e-4, 0.0), ("c", 0.0, 4.0e-4), *far]
        report = near_point_pairs(points, 1.0e-3, max_pairs=2)
        self.assertEqual(report["count"], 3)
        self.assertEqual(len(report["pairs"]), 2)
        labels = {item["a"] for item in report["pairs"]} | {item["b"] for item in report["pairs"]}
        self.assertLessEqual(labels, {"a", "b", "c"})

    def test_dwg_converter_module_formats_command_templates(self) -> None:
        cmd = dwg_converter_command(["dwgtool", "--input={input}", "--output={output}"], Path("a.dwg"), Path("b.dxf"))
        self.assertEqual(cmd, ["dwgtool", "--input=a.dwg", "--output=b.dxf"])

    def test_gf1_payload_module_extracts_compressed_json_text(self) -> None:
        payload = b'prefix' + zlib.compress(b'{"geometry":{"lines":[{"start":[0,0],"end":[1,0]}]}}')
        candidates = gf1_payload_text_candidates(payload)
        self.assertTrue(any('"geometry"' in text for _label, text in candidates))
        self.assertEqual(best_effort_decode(b"plain text"), "plain text")


class GuiModelCheckTests(unittest.TestCase):
    def test_model_check_reports_preflight_risks(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui
        from geofem_app.gui.model_check_worker import collect_model_issues_snapshot

        issues_holder: list[list[tuple[str, str, str, dict[str, object]]]] = []
        original_exec = QApplication.exec

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u"]},
                "mesh": {
                    "nodes": {
                        "1": [0.0, 0.0],
                        "2": [1.0, 0.0],
                        "3": [1.0, 1.0],
                        "4": [0.0, 1.0],
                        "5": [4.0, 4.0],
                    },
                    "elements": [
                        {"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"},
                        {"id": "2", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "missing"},
                    ],
                },
                "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
                "boundary_conditions": [{"node": "1", "ux": 0.0}, {"node": "1", "ux": 1.0}, {"node": "2", "ux": 0.0}],
                "loads": [{"node": "1", "fx": 10.0}, {"edge": ["1", "2"], "tx": 3.0}],
                "stages": [
                    {
                        "name": "A",
                        "type": "excavation",
                        "elements": ["1"],
                        "stress_release": 1.5,
                        "hydro": {"pressure_bcs": [{"node": "99", "value": 0.0}]},
                    },
                    {"name": "A", "type": "death", "elements": ["1"], "loads": [{"node": "1", "fx": 5.0}]},
                ],
                "checks": {"mesh_quality": {"min_area": 2.0, "min_angle_deg": 100.0, "max_aspect_ratio": 0.5}},
            }
            sync_issues = window.collect_model_issues()
            detached_issues = collect_model_issues_snapshot(type(window), window.cfg)
            self.assertEqual([issue[:3] for issue in detached_issues], [issue[:3] for issue in sync_issues])
            issues_holder.append(detached_issues)
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec

        self.assertEqual(len(issues_holder), 1)
        issues = issues_holder[0]

        def has(severity: str, target_prefix: str, text: str | None = None) -> bool:
            return any(
                item_severity == severity and target.startswith(target_prefix) and (text is None or text in detail)
                for item_severity, target, detail, _payload in issues
            )

        self.assertTrue(has("ERROR", "materials", "未定義材料"))
        self.assertTrue(has("ERROR", "mesh.duplicate_elements"))
        self.assertTrue(has("ERROR", "mesh.quality.area"))
        self.assertTrue(has("WARN", "mesh.quality.angle"))
        self.assertTrue(has("WARN", "mesh", "未接続節点"))
        self.assertTrue(has("ERROR", "boundary_conditions", "矛盾"))
        self.assertTrue(has("WARN", "boundary_conditions.rigid_modes"))
        self.assertTrue(has("WARN", "loads", "拘束済みDOF"))
        self.assertTrue(has("WARN", "loads[2]", "分布荷重"))
        self.assertTrue(has("WARN", "stages", "重複"))
        self.assertTrue(has("ERROR", "stages[1]", "stress_release"))
        self.assertTrue(has("WARN", "stages[2]", "非アクティブ"))
        self.assertTrue(has("ERROR", "stages[1].hydro.pressure_bcs"))

    def test_axisymmetric_gui_presets_guides_legend_and_model_check(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "axisymmetric_static", "geometry": "axisymmetric", "fields": ["u", "p"]},
                "mesh": {
                    "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                    "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    "node_sets": {},
                },
                "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
                "stages": [{"name": "Axisym-up", "type": "consolidation"}],
            }
            window.populate_forms()
            window.apply_axisymmetric_reference_sets()
            node_sets = window.cfg["mesh"]["node_sets"]
            self.assertEqual(node_sets["axisymmetric_axis"], ["1", "4"])
            self.assertEqual(node_sets["axisymmetric_outer_radius"], ["2", "3"])

            window.axisym_boundary_preset.setCurrentIndex(window.axisym_boundary_preset.findData("minimal_support"))
            window.apply_axisymmetric_boundary_preset()
            self.assertTrue(any(bc.get("support_type") == "axisymmetric_symmetry_axis" for bc in window.cfg["boundary_conditions"]))

            window.axisym_load_preset.setCurrentIndex(window.axisym_load_preset.findData("outer_radial_pressure"))
            window.axisym_load_value.setText("12.5")
            window.apply_axisymmetric_load_preset()
            self.assertEqual(window.cfg["loads"][-1]["component"], "radial_surface_pressure")
            self.assertEqual(window.cfg["loads"][-1]["coordinate_system"], "axisymmetric_rz")

            window.axisym_hydro_preset.setCurrentIndex(window.axisym_hydro_preset.findData("outer_robin"))
            window.axisym_hydro_value.setText("4.0")
            window.axisym_hydro_beta.setText("0.75")
            window.apply_axisymmetric_hydro_preset()
            hydro = window.cfg["stages"][0]["hydro"]
            self.assertEqual(hydro["pore_robin_bcs"][-1]["component"], "outer_robin_drain")

            issues = window.collect_model_issues()
            self.assertTrue(any(target == "axisymmetric.gui" for _sev, target, _detail, _payload in issues))
            self.assertTrue(any(target == "axisymmetric.boundary" for _sev, target, _detail, _payload in issues))

            window.result_element_values = {"1": 1.5}
            window.post_mode = "contour"
            window.post_component = "sigma_r"
            window.update_preview()
            scene_data = [item.data(0) for item in window.scene.items() if isinstance(item.data(0), dict)]
            self.assertTrue(any(data.get("kind") == "axisymmetric_radial_guide" for data in scene_data))
            self.assertTrue(any(data.get("kind") == "axisymmetric_radius_label" for data in scene_data))
            self.assertTrue(any(data.get("kind") == "axisymmetric_legend_note" for data in scene_data))
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_cad_mesh_operation_ui_round_trips_controls(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        line_a = {"type": "line", "start": [0.0, 0.0], "end": [2.0, 0.0]}
        line_b = {"type": "line", "start": [1.0, 0.0], "end": [3.0, 0.0]}
        overlap_graph = classify_graph_boolean_operations(build_analytic_curve_boolean_graph([[line_a], [line_b]], target=0.1, tol=1.0e-9))
        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain"},
                "mesh": {
                    "nodes": {"1": [0.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [0.0, 1.0]},
                    "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    "refinements": [{"id": "r1", "center": [1.0, 0.5], "radius": 0.25, "factor": 2.0}],
                    "control_points": [{"id": "cp1", "point": [1.0, 0.5], "target_size": 0.2, "tag": "crown"}],
                    "split_lines": [{"id": "s1", "type": "split_line", "start": [1.0, 0.0], "end": [1.0, 1.0], "target_size": 0.25, "locked": True}],
                    "size_map": [{"id": "sm1", "center": [1.5, 0.5], "radius": 0.4, "target_size": 0.25, "grading": 1.3}],
                    "blocks": {"B1": {"id": "B1", "name": "B1", "elements": ["1"], "active": True, "split_hint": "vertical"}},
                    "cad_boolean": {"analytic_curve_graph": overlap_graph},
                },
                "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
                "checks": {"mesh_quality": {"min_area": 2.0, "min_angle_deg": 100.0, "max_aspect_ratio": 0.5}},
                "geometry": {
                    "layers": [
                        {"name": "CAD", "visible": True, "color": "#7a5c00", "linetype": "continuous", "source": "test"},
                        {"name": "Hidden", "visible": False, "color": "#cccccc", "source": "test"},
                    ],
                    "lines": [
                        {"id": "L1", "layer": "CAD", "purpose": "helper", "start": [0.0, 0.0], "end": [2.0, 0.0]},
                        {"id": "L2", "layer": "Hidden", "purpose": "helper", "start": [0.0, 1.0], "end": [2.0, 1.0]},
                    ],
                    "annotations": [{"id": "N1", "layer": "CAD", "point": [0.5, 0.2], "text": "note"}],
                    "dimensions": [{"id": "D1", "layer": "CAD", "start": [0.0, 0.0], "end": [2.0, 0.0], "text": "2.0m"}],
                    "cad_boolean": {"analytic_curve_graph": overlap_graph},
                },
            }
            window.populate_forms()
            self.assertEqual(window.geometry_layer_table.rowCount(), 2)
            self.assertEqual(window.geometry_annotation_table.rowCount(), 1)
            self.assertEqual(window.geometry_dimension_table.rowCount(), 1)
            self.assertGreaterEqual(window.geometry_overlap_table.rowCount(), 2)
            self.assertEqual(window.mesh_refinement_table.rowCount(), 1)
            self.assertEqual(window.mesh_control_point_table.rowCount(), 1)
            self.assertEqual(window.mesh_split_line_table.rowCount(), 1)
            self.assertEqual(window.mesh_size_map_table.rowCount(), 1)
            self.assertEqual(window.mesh_block_table.rowCount(), 1)
            self.assertGreaterEqual(window.mesh_quality_violation_table.rowCount(), 1)
            window._refresh_tree(select_panel="mesh")
            QApplication.processEvents()
            root = window.tree.topLevelItem(0)
            quality_item = window._find_tree_panel_item(root, "mesh_quality:element:1")
            self.assertIsNotNone(quality_item)
            window.tree.setCurrentItem(quality_item)
            QApplication.processEvents()
            self.assertIn("1", window._selected_preview_entities()["elements"])

            window.apply_geometry_panel()
            layers = {layer["name"]: layer for layer in window.cfg["geometry"]["layers"]}
            self.assertFalse(layers["Hidden"]["visible"])
            self.assertEqual(window.cfg["geometry"]["annotations"][0]["text"], "note")
            self.assertEqual(window.cfg["geometry"]["dimensions"][0]["text"], "2.0m")

            window.snap_type.setCurrentIndex(window.snap_type.findData("grid"))
            window.snap_grid_size.setText("0.5")
            snapped = window._snap_model_point(0.74, 1.26)
            self.assertEqual(snapped, (0.5, 1.5, "grid:0.5"))

            window.apply_mesh_controls_panel()
            self.assertEqual(window.cfg["mesh"]["control_points"][0]["tag"], "crown")
            self.assertEqual(window.cfg["mesh"]["split_lines"][0]["id"], "s1")
            self.assertEqual(window.cfg["mesh"]["size_map"][0]["id"], "sm1")
            self.assertEqual(window.cfg["mesh"]["blocks"]["B1"]["split_hint"], "vertical")
            window.snap_enabled.setChecked(False)
            window.update_preview()
            window.begin_mesh_control_drag({"kind": "mesh_control_point", "id": "cp1"})
            scene_point = QPointF(1.25 * window.preview_scale + window.preview_ox, window.preview_oy - 0.25 * window.preview_scale)
            window.update_mesh_control_drag(scene_point, final=True)
            self.assertEqual(window.cfg["mesh"]["control_points"][0]["point"], [1.25, 0.25])

            window.mesh_quality_violation_table.selectRow(0)
            window.select_mesh_quality_violations()
            QApplication.processEvents()
            current_tree_key = str(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole))
            self.assertEqual(current_tree_key, "mesh_quality:element:1")
            selected = [item.data(0) for item in window.scene.selectedItems() if isinstance(item.data(0), dict)]
            self.assertTrue(any(data.get("kind") in {"element", "mesh_quality_violation"} for data in selected))
            self.assertTrue(any(data.get("kind") == "node" for data in selected))
            window.snap_enabled.setChecked(False)
            window.begin_mesh_node_drag({"kind": "node", "id": "2"})
            scene_point = QPointF(2.15 * window.preview_scale + window.preview_ox, window.preview_oy - 0.1 * window.preview_scale)
            window.update_mesh_node_drag(scene_point, final=False)
            window.update_mesh_node_drag(scene_point, final=True)
            self.assertEqual(window.cfg["mesh"]["nodes"]["2"], [2.15, 0.1])
            self.assertTrue(window.cfg["mesh"].get("manual_node_moves"))
            window.repair_selected_mesh_quality_violations()
            self.assertTrue(window.cfg["mesh"].get("quality_repairs"))
            self.assertTrue(any(raw.get("source") == "quality_violation" for raw in window.cfg["mesh"].get("refinements", []) if isinstance(raw, dict)))
            self.assertTrue(any(raw.get("source") == "quality_violation" for raw in window.cfg["mesh"].get("control_points", []) if isinstance(raw, dict)))
            self.assertTrue(any(raw.get("source") == "quality_violation" for raw in window.cfg["mesh"].get("size_map", []) if isinstance(raw, dict)))
            self.assertTrue(window.cfg["mesh"].get("requires_rebuild"))
            window.mesh_quality_method.setCurrentIndex(window.mesh_quality_method.findData("all"))
            window.mesh_quality_iterations.setText("3")
            window.compare_mesh_quality_improvements()
            self.assertEqual(window.mesh_quality_improvement_table.rowCount(), 4)
            element_count_before_improve = len(window.cfg["mesh"]["elements"])
            window.mesh_quality_improvement_table.selectRow(2)
            window.apply_selected_mesh_quality_improvement()
            self.assertGreater(len(window.cfg["mesh"]["elements"]), element_count_before_improve)
            self.assertTrue(window.cfg["mesh"].get("quality_improvement_history"))

            window.update_preview()
            for item in window.scene.items():
                data = item.data(0)
                if isinstance(data, dict) and data.get("kind") == "element" and data.get("id") == "1":
                    item.setSelected(True)
                    break
            before_split_count = len(window.cfg["mesh"].get("split_lines", []))
            window.split_selected_mesh_block()
            self.assertGreater(len(window.cfg["mesh"].get("split_lines", [])), before_split_count)
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_material_advanced_forms_write_solver_connected_models(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            self.assertGreaterEqual(window.material_library_model.minimumWidth(), 280)
            self.assertGreaterEqual(window.material_library_model.view().minimumWidth(), 360)
            index = window.material_library_model.findData("pastor_zienkiewicz_sand")
            self.assertGreaterEqual(index, 0)
            window.material_library_model.setCurrentIndex(index)
            window.material_test_curve.setPlainText(
                "curve,gamma%,G\n"
                "CD,0.01,30000\n"
                "CD,0.1,15000\n"
                "CU,0.01,28000\n"
                "CU,0.1,14000\n"
            )
            window.estimate_material_constants_from_curve()
            window.material_river_n.setText("15")
            window.material_river_fc.setText("20")
            window.material_river_sigma_v.setText("100")
            window.material_river_sigma_v_eff.setText("80")
            window.material_river_csr.setText("0.18")
            window.apply_river_seismic_parameters()
            window.add_material_from_library()
            window.apply_materials_panel()
            material = window.cfg["materials"]["sand_pz"]
            self.assertEqual(material["model"], "pastor_zienkiewicz_sand")
            self.assertIn("test_curve", material)
            self.assertIn("curve_fits", material)
            self.assertIn("fit_confidence", material)
            self.assertIn("fit_report", material)
            self.assertIn("fit_report_html", material)
            self.assertIn("global_fit", material)
            self.assertEqual(material["unit_conversion"]["gamma_scale"], 0.01)
            self.assertEqual(set(material["curve_fits"]), {"CD", "CU"})
            self.assertIn("river_seismic_guideline", material)
            self.assertIn("liquefaction", material)
            checks.append({"ok": True, "G0": material["G0"]})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(len(checks), 1)
        self.assertGreater(float(checks[0]["G0"]), 0.0)

    def test_curve_boolean_gui_rebuilds_analytic_graph_from_curve_table(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF
            from PySide6.QtWidgets import QApplication, QTableWidgetItem
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        region_1 = {
            "id": "A",
            "curve_boundary": [
                {"type": "line", "start": [0.0, 0.0], "end": [2.0, 0.0]},
                {"type": "line", "start": [2.0, 0.0], "end": [2.0, 1.0]},
                {
                    "type": "nurbs",
                    "control_points": [[2.0, 1.0], [1.4, 1.3], [0.6, 1.3], [0.0, 1.0]],
                    "weights": [1.0, 1.0, 1.0, 1.0],
                    "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    "degree": 3,
                },
                {"type": "line", "start": [0.0, 1.0], "end": [0.0, 0.0]},
            ],
        }
        region_2 = {
            "id": "B",
            "curve_boundary": [
                {"type": "line", "start": [1.0, 0.0], "end": [3.0, 0.0]},
                {"type": "line", "start": [3.0, 0.0], "end": [3.0, 1.0]},
                {"type": "bezier", "control_points": [[3.0, 1.0], [2.4, 1.25], [1.6, 1.25], [1.0, 1.0]]},
                {"type": "line", "start": [1.0, 1.0], "end": [1.0, 0.0]},
            ],
        }
        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain"},
                "mesh": {
                    "nodes": {"1": [0.0, 0.0], "2": [3.0, 0.0], "3": [3.0, 1.5], "4": [0.0, 1.5]},
                    "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    "target_size": 0.2,
                    "boolean_expression": "A-B",
                },
                "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
                "geometry": {"regions": [region_1, region_2], "boolean_expression": "A-B"},
            }
            window.populate_forms()
            self.assertEqual(window.geometry_curve_table.rowCount(), 8)
            curve_types = {window._table_text(window.geometry_curve_table, row, 2) for row in range(window.geometry_curve_table.rowCount())}
            self.assertIn("nurbs", curve_types)
            self.assertIn("bezier", curve_types)
            self.assertGreaterEqual(window.geometry_curve_control_table.rowCount(), 8)

            for row in range(window.geometry_curve_control_table.rowCount()):
                curve_row = window._table_text(window.geometry_curve_control_table, row, 0)
                point = window._table_text(window.geometry_curve_control_table, row, 1)
                if curve_row == "3" and point == "2":
                    window.geometry_curve_control_table.setItem(row, 4, QTableWidgetItem("1.45"))
                    break
            window.apply_curve_control_point_table()
            self.assertAlmostEqual(window.cfg["geometry"]["regions"][0]["curve_boundary"][2]["control_points"][1][1], 1.45)
            window.populate_forms()
            window.snap_enabled.setChecked(False)
            window.update_preview()
            window.begin_cad_handle_drag({"kind": "curve_control_point", "region": 1, "role": "boundary", "curve_role": "boundary", "control_role": "control", "curve": 3, "point": 2})
            window.update_cad_handle_drag(QPointF(1.4 * window.preview_scale + window.preview_ox, window.preview_oy - 1.6 * window.preview_scale), final=True)
            self.assertAlmostEqual(window.cfg["geometry"]["regions"][0]["curve_boundary"][2]["control_points"][1][1], 1.6)
            self.assertTrue(window.cfg["geometry"].get("cad_boolean", {}).get("graph_stale", False) or "cad_boolean" not in window.cfg["geometry"])

            window.add_geometry_layer_row(name="CAD_edit", visible=True, locked=True, color="#123456", lineweight="250", opacity="0.5", source="gui")
            window.add_geometry_dimension_row(dimension_id="D_lock", layer="CAD_edit", x1="0", y1="0", x2="2", y2="0", text="2.5", constraint="length", locked=True)
            window.cfg["geometry"]["cad_constraints"] = [
                {
                    "id": "G1_bottom_lines",
                    "type": "tangent",
                    "points": ["curve:1:boundary:1:start", "curve:1:boundary:1:end"],
                    "reference_points": ["curve:2:boundary:1:start", "curve:2:boundary:1:end"],
                }
            ]
            window.apply_dimension_constraints()
            layers = {layer["name"]: layer for layer in window.cfg["geometry"]["layers"]}
            self.assertTrue(layers["CAD_edit"]["locked"])
            self.assertAlmostEqual(layers["CAD_edit"]["lineweight"], 250.0)
            self.assertEqual(window.cfg["geometry"]["dimension_constraints"][0]["type"], "length")
            self.assertEqual(window.cfg["geometry"]["dimension_solver"]["engine"], "cad_constraint_least_squares")
            self.assertFalse(window.cfg["geometry"]["dimension_solver"]["inconsistent"])
            self.assertEqual(window.cfg["geometry"]["dimension_solver"]["explicit_constraint_count"], 1)
            self.assertTrue(any(row["label"] == "G1_bottom_lines:tangent" for row in window.cfg["geometry"]["dimension_solver"]["residuals"]))
            self.assertAlmostEqual(window.cfg["geometry"]["regions"][0]["curve_boundary"][0]["end"][0], 2.5)
            self.assertAlmostEqual(window.cfg["geometry"]["regions"][0]["curve_boundary"][1]["start"][0], 2.5)
            window.populate_forms()

            for row in range(window.geometry_curve_table.rowCount()):
                if window._table_text(window.geometry_curve_table, row, 2) == "nurbs":
                    window.geometry_curve_table.setItem(row, 5, QTableWidgetItem("12"))
                    break
            window.geometry_boolean_expression.setText("A-B")
            window.geometry_boolean_operation.setCurrentText("expression")
            window.rebuild_geometry_boolean_graph()
            cad_boolean = window.cfg["geometry"]["cad_boolean"]
            graph = cad_boolean["analytic_curve_graph"]
            self.assertEqual(cad_boolean["engine"], "analytic_curve_graph_winding_containment")
            self.assertEqual(graph["boolean_expression"]["normalized"], "(R1-R2)")
            self.assertGreater(graph["boolean_operations"]["expression"]["edge_count"], 0)
            self.assertGreaterEqual(window.geometry_overlap_table.rowCount(), 2)
            self.assertGreaterEqual(window.geometry_boolean_table.rowCount(), 5)
            for row in range(window.geometry_boolean_table.rowCount()):
                if window._table_text(window.geometry_boolean_table, row, 0) == "expression":
                    window.geometry_boolean_table.selectRow(row)
                    break
            window.select_cad_boolean_operation_from_table()
            window.store_selected_boolean_operation_as_manual_repair()
            self.assertEqual(window.cfg["geometry"]["cad_boolean"]["manual_selected_operation"], "expression")
            self.assertIn("expression", window.cfg["geometry"]["cad_boolean"]["manual_loop_overrides"])
            window.update_preview()
            self.assertTrue(any((item.data(0) or {}).get("kind") == "cad_boolean_loop" for item in window.scene.items() if isinstance(item.data(0), dict)))

            window.geometry_overlap_table.selectRow(0)
            first_edge = window._table_text(window.geometry_overlap_table, 0, 0)
            window.set_selected_cad_overlap_state("suppressed")
            self.assertEqual(window.cfg["geometry"]["cad_boolean"]["trim_edge_overrides"][first_edge], "suppressed")
            window.geometry_overlap_table.selectRow(0)
            window.repair_selected_cad_overlap_edges()
            self.assertEqual(window.cfg["geometry"]["cad_boolean"]["trim_edge_overrides"][first_edge], "repaired_prefer_primary")
            self.assertIn(first_edge, window.cfg["geometry"]["cad_boolean"]["manual_edge_repairs"])
            self.assertEqual(window.cfg["geometry"]["regions"][0]["curve_boundary"][2]["segments"], 12)
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_stage_analysis_gui_uses_detailed_geo_feas_like_forms(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QPushButton
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
                "mesh": {
                    "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                    "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    "element_sets": {"excavation_zone": ["1"]},
                    "node_sets": {"top": ["4", "3"], "bottom": ["1", "2"]},
                },
                "materials": {
                    "soil": {"model": "elastic", "E": 1000.0, "nu": 0.3},
                    "liner": {"model": "elastic", "E": 2000.0, "nu": 0.25},
                },
                "stages": [{"name": "Stage-1", "type": "static"}],
            }
            window.populate_forms()
            window.stage_table.selectRow(0)
            window.populate_stage_detail_tables()
            self.assertEqual(window.stage_table.columnCount(), 10)
            self.assertEqual(window.stage_change_table.columnCount(), 13)
            self.assertIn("推奨値", window.stage_recommendation_label.text())
            self.assertEqual(window.stage_detail_increments.text(), "1")
            detail_index = window.gui_operation_mode_combo.findData("detail")
            window.gui_operation_mode_combo.setCurrentIndex(detail_index)
            QApplication.processEvents()
            window._activate_panel("stages")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            self.assertTrue(window.inline_detail_sheet_group.isVisible())
            self.assertIs(window.inline_detail_stack.currentWidget(), window.panel_pages["stages"])
            self.assertTrue(window.aux_stage_action_group.isVisible())
            self.assertTrue(window.aux_stage_inspector_group.isVisible())
            self.assertGreaterEqual(window.inline_detail_sheet_group.maximumHeight(), 560)
            window._refresh_tree(select_panel="stage:item:1")
            self.assertEqual(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole), "stage:item:1")
            self.assertIs(window.tabs.currentWidget(), window.model_workspace_page)
            sheet_buttons = [button for button in window.panel_pages["stages"].findChildren(QPushButton) if button.property("stageSheetButton")]
            list_buttons = [button for button in sheet_buttons if button.property("stageSheetButtonGroup") == "stage_list"]
            change_buttons = [button for button in sheet_buttons if button.property("stageSheetButtonGroup") == "stage_change"]
            self.assertFalse(list_buttons)
            self.assertGreaterEqual(len(change_buttons), 7)
            self.assertTrue(all(button.minimumWidth() >= 120 for button in change_buttons))
            self.assertEqual({button.property("uniformPanelButtonColumns") for button in change_buttons}, {4})
            right_pane_stage_buttons = [
                button
                for button in window.aux_stage_action_group.findChildren(QPushButton)
                if button.property("stageSheetButtonGroup") == "stage_list"
            ]
            self.assertGreaterEqual(len(right_pane_stage_buttons), 6)
            self.assertTrue(all(button.minimumWidth() >= 120 for button in right_pane_stage_buttons))
            self.assertEqual({button.property("uniformPanelButtonColumns") for button in right_pane_stage_buttons}, {2})
            self.assertFalse(window.stage_template_save_button.isVisible())
            self.assertFalse(window.stage_template_load_button.isVisible())
            inline_detail_buttons = [
                button
                for button in window.stage_form_workspace.findChildren(QPushButton)
                if button.property("stageDetailSheetButton")
            ]
            self.assertFalse(inline_detail_buttons)
            right_pane_detail_buttons = [
                button
                for button in window.aux_stage_inspector_group.findChildren(QPushButton)
                if button.property("stageDetailSheetRightPaneButton")
            ]
            self.assertGreaterEqual(len(right_pane_detail_buttons), 17)
            self.assertNotIn("材料変更追加", {button.text() for button in right_pane_detail_buttons})
            self.assertTrue(all(button.minimumWidth() >= 120 for button in right_pane_detail_buttons))
            self.assertEqual({button.property("uniformPanelButtonColumns") for button in right_pane_detail_buttons}, {2})
            window.stage_detail_tabs.setCurrentIndex(0)
            window._refresh_stage_detail_sheet_action_buttons()
            self.assertFalse(window.stage_detail_sheet_action_group.isVisible())
            window.stage_detail_tabs.setCurrentIndex(1)
            window._refresh_stage_detail_sheet_action_buttons()
            self.assertTrue(window.stage_detail_sheet_action_group.isVisible())
            self.assertIn("境界", window.stage_detail_sheet_action_group.title())
            window.stage_detail_type.setCurrentText("srm")
            window._refresh_stage_inspector_visibility()
            srm_fields = [
                wrapper
                for wrapper in window.stage_inspector_field_wrappers
                if wrapper.property("stageInspectorFamily") == "srm"
            ]
            self.assertTrue(srm_fields)
            self.assertTrue(all(wrapper.isVisible() for wrapper in srm_fields))
            self.assertFalse(any(wrapper.isVisible() for wrapper in window.stage_inspector_field_wrappers if wrapper.property("stageInspectorFamily") == "geostatic"))
            window.apply_stage_template_from_library("斜面安定:SRM照査")
            self.assertEqual(window.cfg["stages"][0]["srm"]["factor_max"], 3.0)
            window.populate_stage_detail_tables()
            self.assertEqual(window.stage_detail_srm_end.text(), "3.0")
            from geofem_app.fem2d_solver_controls import srm_factors

            self.assertEqual(srm_factors({"start": 1.0, "end": 1.1, "step": 0.1}), [1.0, 1.1])
            window.stage_detail_type.setCurrentText("death")
            window.stage_detail_target.clear()
            window.stage_detail_stress_release.clear()
            window.stage_detail_increments.clear()
            window.stage_detail_solver.setPlainText("")
            window.apply_stage_recommended_defaults(force=True)
            self.assertEqual(window.stage_detail_target.text(), "all")
            self.assertEqual(window.stage_detail_stress_release.text(), "1.0")
            self.assertEqual(window.stage_detail_increments.text(), "4")
            self.assertIn("cutback", window.stage_detail_solver.toPlainText())

            window.stage_detail_name.setText("Excavation-1")
            window.stage_detail_type.setCurrentText("consolidation")
            window.stage_detail_target.setText("excavation_zone")
            window.stage_detail_stress_release.setText("0.65")
            window.stage_detail_k0.setText("0.55")
            window.stage_detail_surface_y.setText("1.0")
            window.stage_detail_dt.setText("0.25")
            window.stage_detail_steps.setText("4")
            window.stage_detail_storage.setText("0.02")
            window.stage_detail_permeability.setText("1e-5")
            window.stage_detail_biot_alpha.setText("0.9")
            window.stage_detail_increments.setText("3")
            window.stage_detail_solver.setPlainText("max_iter: 24\ntolerance: 1.0e-6\n")
            window.add_stage_construction_row(action="excavation", target="excavation_zone", stress_release="0.65")
            window.add_stage_construction_row(action="material", target="excavation_zone", material="liner")
            window.add_stage_hydro_row(kind="pressure", target="top", value="10.0")
            window.add_stage_hydro_row(kind="flux", target="bottom", value="-0.01")
            window.add_stage_mpc_row(master="1", slave="2", dof="ux", coefficient="1.0", value="0.0", method="lagrange")
            window.apply_stage_detail_tables()

            stage = window.cfg["stages"][0]
            self.assertEqual(stage["name"], "Excavation-1")
            self.assertEqual(stage["type"], "excavation")
            self.assertEqual(stage["set"], "excavation_zone")
            self.assertEqual(stage["stress_release"], 0.65)
            self.assertEqual(stage["k0"], 0.55)
            self.assertEqual(stage["surface_y"], 1.0)
            self.assertEqual(stage["hydro"]["dt"], 0.25)
            self.assertEqual(stage["hydro"]["steps"], 4)
            self.assertEqual(stage["hydro"]["storage"], 0.02)
            self.assertEqual(stage["hydro"]["permeability"], 1.0e-5)
            self.assertEqual(stage["hydro"]["biot_alpha"], 0.9)
            self.assertEqual(len(stage["hydro"]["pressure_bcs"]), 1)
            self.assertEqual(len(stage["hydro"]["pore_flux_bcs"]), 1)
            self.assertEqual(stage["element_properties"][0]["material"], "liner")
            self.assertEqual(stage["mpc_constraints"][0]["method"], "lagrange")
            self.assertEqual(stage["solver"]["max_iter"], 24)
            self.assertEqual(stage["increments"], 3)
            self.assertGreaterEqual(len(stage["construction_events"]), 2)
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_srm_post_gui_builds_geo_feas_like_slip_candidates(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain"},
                "mesh": {
                    "nodes": {
                        "1": [0.0, 0.0],
                        "2": [1.0, 0.0],
                        "3": [2.0, 0.0],
                        "4": [0.0, 1.0],
                        "5": [1.0, 1.0],
                        "6": [2.0, 1.0],
                        "7": [0.0, 2.0],
                        "8": [1.0, 2.0],
                        "9": [2.0, 2.0],
                    },
                    "elements": [
                        {"id": "1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil"},
                        {"id": "2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil"},
                        {"id": "3", "type": "QUAD4", "nodes": ["4", "5", "8", "7"], "material": "soil"},
                        {"id": "4", "type": "QUAD4", "nodes": ["5", "6", "9", "8"], "material": "soil"},
                    ],
                },
                "materials": {"soil": {"model": "mohr_coulomb", "E": 1000.0, "nu": 0.3, "cohesion": 10.0, "phi": 30.0}},
                "stages": [
                    {
                        "name": "SRM",
                        "type": "srm",
                        "solver": {
                            "srm": {
                                "factor_of_safety": 1.18,
                                "trials": [
                                    {"factor": 1.0, "converged": True, "plastic_ratio": 0.25, "ok": True},
                                    {"factor": 1.2, "converged": False, "plastic_ratio": 1.0, "ok": False, "note": "failure"},
                                ],
                            }
                        },
                    }
                ],
            }
            window.populate_forms()
            window.result_rows = [
                {"element_id": "1", "ip": "1", "material": "soil", "active": "1", "q": "50.0", "p": "100.0", "plastic": "1.0", "FL": "0.82"},
                {"element_id": "1", "ip": "2", "material": "soil", "active": "1", "q": "42.0", "p": "90.0", "plastic": "0.7", "FL": "0.92"},
                {"element_id": "2", "ip": "1", "material": "soil", "active": "1", "q": "48.0", "p": "95.0", "plastic": "1.0", "FL": "0.88"},
                {"element_id": "2", "ip": "2", "material": "soil", "active": "1", "q": "40.0", "p": "92.0", "plastic": "0.6", "FL": "0.95"},
                {"element_id": "3", "ip": "1", "material": "soil", "active": "1", "q": "46.0", "p": "93.0", "plastic": "1.0", "FL": "0.86"},
                {"element_id": "4", "ip": "1", "material": "soil", "active": "1", "q": "20.0", "p": "80.0", "plastic": "0.0", "FL": "1.35"},
            ]

            window.show_srm_post()

            self.assertEqual(window.post_mode, "srm")
            self.assertEqual(window.post_component, "FL")
            self.assertEqual(window.srm_trial_table.rowCount(), 2)
            self.assertGreaterEqual(window.srm_slip_table.rowCount(), 1)
            self.assertGreaterEqual(window.srm_candidate_compare_table.rowCount(), 2)
            self.assertGreaterEqual(window.srm_local_fl_table.rowCount(), 6)
            self.assertIn("1", window._table_text(window.srm_slip_table, 0, 1))
            self.assertIn("2", window._table_text(window.srm_slip_table, 0, 1))
            self.assertIn("SRM安全率", window.srm_post_summary.text())
            candidate_types = {window._table_text(window.srm_candidate_compare_table, row, 1) for row in range(window.srm_candidate_compare_table.rowCount())}
            self.assertIn("circular arc", candidate_types)
            self.assertTrue(any("optimized" in value for value in candidate_types))
            self.assertAlmostEqual(window.result_element_values["1"], 0.87)
            window.srm_local_fl_aggregation.setCurrentText("min")
            window.show_srm_post(update_only=True)
            self.assertAlmostEqual(window.result_element_values["1"], 0.82)
            window.srm_slope_direction.setCurrentText("right-to-left")
            window.srm_min_candidate_length.setText("1.0")
            window.srm_max_circle_radius.setText("10.0")
            window.srm_require_boundary_exit.setChecked(True)
            window.show_srm_post(update_only=True)
            self.assertTrue(window.srm_slip_candidates)
            constrained = window.srm_slip_candidates[0]
            self.assertEqual(constrained["engineering_constraints"]["direction"], "right-to-left")
            self.assertGreaterEqual(constrained["length"], 1.0)
            self.assertGreaterEqual(constrained["points"][0][0], constrained["points"][-1][0])
            report = window.build_srm_candidate_report(Path(tempfile.gettempdir()) / "srm_candidate_report_test.html")
            self.assertIsNotNone(report)
            self.assertTrue(report.exists())
            self.assertIn("SRM", report.read_text(encoding="utf-8"))
            slip_items = [item for item in window.scene.items() if isinstance(item.data(0), dict) and item.data(0).get("kind") == "srm_slip_candidate"]
            self.assertTrue(slip_items)
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_boundary_gui_batch_selection_creates_support_hydro_and_mpc(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u"]},
                "mesh": {
                    "nodes": {
                        "1": [0.0, 0.0],
                        "2": [1.0, 0.0],
                        "3": [1.0, 1.0],
                        "4": [0.0, 1.0],
                    },
                    "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                },
                "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
                "stages": [{"name": "Stage-1", "type": "static"}],
            }
            window.populate_forms()
            window.update_preview()
            window.stage_table.selectRow(0)

            window._select_tree_panel("boundary_conditions")
            window.activate_boundary_selection_tool()
            window.show()
            QApplication.processEvents()

            def real_click_model_item(kind: str, ident: str) -> None:
                target = next(
                    item
                    for item in window.scene.items()
                    if isinstance(item.data(0), dict)
                    and item.data(0).get("kind") == kind
                    and str(item.data(0).get("id")) == ident
                )
                pos = window.view.mapFromScene(target.sceneBoundingRect().center())
                press = QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress,
                    QPointF(pos),
                    QPointF(window.view.viewport().mapToGlobal(pos)),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                release = QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease,
                    QPointF(pos),
                    QPointF(window.view.viewport().mapToGlobal(pos)),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                window.view.mousePressEvent(press)
                window.view.mouseReleaseEvent(release)
                QApplication.processEvents()

            def highlight_target_kinds() -> set[str]:
                return {
                    str(item.data(0).get("target_kind"))
                    for item in window.scene.items()
                    if isinstance(item.data(0), dict) and item.data(0).get("kind") == "selection_highlight"
                }

            real_click_model_item("node", "1")
            self.assertEqual(window._selected_preview_entities()["nodes"], {"1"})
            self.assertIn("node", highlight_target_kinds())
            real_click_model_item("edge", "1-2")
            self.assertEqual(window._selected_preview_entities()["edges"], [("1", "2")])
            self.assertIn("edge", highlight_target_kinds())
            window.scene.clearSelection()
            self.assertFalse(highlight_target_kinds())
            edge_item = next(
                item
                for item in window.scene.items()
                if isinstance(item.data(0), dict)
                and item.data(0).get("kind") == "edge"
                and str(item.data(0).get("id")) == "1-2"
            )
            right_click_pos = window.view.mapFromScene(edge_item.sceneBoundingRect().center())
            context_calls: list[str] = []
            original_context_menu = window._show_command_context_menu
            window._show_command_context_menu = lambda _global_pos, context: context_calls.append(context)
            try:
                window.view.contextMenuEvent(
                    type(
                        "FakeBoundaryContextMenuEvent",
                        (),
                        {
                            "pos": lambda self, p=right_click_pos: p,
                            "globalPos": lambda self, p=window.view.viewport().mapToGlobal(right_click_pos): p,
                            "accept": lambda self: None,
                        },
                    )()
                )
            finally:
                window._show_command_context_menu = original_context_menu
            self.assertEqual(context_calls, ["view"])
            self.assertEqual(window._selected_preview_entities()["edges"], [("1", "2")])
            window.scene.clearSelection()

            window._select_preview_payload({"nodes": ["1", "2"]})
            self.assertIn("節点 2", window.boundary_selection_summary.text())
            window.boundary_batch_scope.setCurrentIndex(window.boundary_batch_scope.findData("global"))
            window.boundary_batch_kind.setCurrentIndex(window.boundary_batch_kind.findData("pin"))
            window.add_selected_boundary_condition()
            self.assertEqual(window.cfg["boundary_conditions"][0]["nodes"], ["1", "2"])
            self.assertTrue(window.cfg["boundary_conditions"][0]["fixed"])
            self.assertEqual(window.boundary_table.rowCount(), 1)
            self.assertEqual(window._selected_preview_entities()["nodes"], {"1", "2"})

            window._select_tree_panel("boundary_conditions")
            QApplication.processEvents()
            def click_model_item(kind: str, ident: str) -> None:
                target = next(
                    item
                    for item in window.scene.items()
                    if isinstance(item.data(0), dict)
                    and item.data(0).get("kind") == kind
                    and str(item.data(0).get("id")) == ident
                )
                pos = window.view.mapFromScene(target.sceneBoundingRect().center())
                window.view.mousePressEvent(
                    type(
                        "FakeSelectClick",
                        (),
                        {
                            "button": lambda self: Qt.MouseButton.LeftButton,
                            "pos": lambda self, p=pos: p,
                            "modifiers": lambda self: Qt.KeyboardModifier.NoModifier,
                            "accept": lambda self: None,
                        },
                    )()
                )
                window.view.mouseReleaseEvent(
                    type(
                        "FakeSelectRelease",
                        (),
                        {
                            "button": lambda self: Qt.MouseButton.LeftButton,
                            "pos": lambda self, p=pos: p,
                            "accept": lambda self: None,
                        },
                    )()
                )
                QApplication.processEvents()

            click_model_item("node", "1")
            self.assertEqual(window._selected_preview_entities()["nodes"], {"1"})
            click_model_item("edge", "1-2")
            self.assertEqual(window._selected_preview_entities()["edges"], [("1", "2")])
            click_model_item("element", "1")
            self.assertEqual(window._selected_preview_entities()["elements"], {"1"})
            context_labels = {label for label, _callback, _tip in window._context_menu_action_specs("view") if label != "-"}
            self.assertIn("固定 ux=uy=0", context_labels)
            self.assertIn("境界表を反映", context_labels)
            window._select_preview_payload({"nodes": ["1"]})
            before_boundary_count = len(window.cfg["boundary_conditions"])
            window.add_selected_boundary_condition_from_context("fixed")
            window.add_selected_boundary_condition_from_context("roller_x")
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["boundary_conditions"]), before_boundary_count + 2)
            self.assertEqual(window.cfg["boundary_conditions"][-2]["nodes"], ["1"])
            self.assertEqual(window.cfg["boundary_conditions"][-1]["nodes"], ["1"])
            self.assertEqual(window.cfg["boundary_conditions"][-1]["ux"], 0.0)
            QApplication.processEvents()
            tree = window.boundary_condition_tree
            node_1_condition_count = 0
            first_node_1_child = None
            for top_index in range(tree.topLevelItemCount()):
                condition_item = tree.topLevelItem(top_index)
                payload = condition_item.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(payload, dict) and "1" in payload.get("nodes", []):
                    node_1_condition_count += 1
                for child_index in range(condition_item.childCount()):
                    child = condition_item.child(child_index)
                    child_payload = child.data(0, Qt.ItemDataRole.UserRole)
                    if isinstance(child_payload, dict) and child_payload.get("nodes") == ["1"] and first_node_1_child is None:
                        first_node_1_child = child
            self.assertGreaterEqual(node_1_condition_count, 2)
            self.assertIsNotNone(first_node_1_child)
            tree.setCurrentItem(first_node_1_child)
            QApplication.processEvents()
            self.assertEqual(window._selected_preview_entities()["nodes"], {"1"})

            root = window.tree.topLevelItem(0)
            left_node_key = "boundary_condition:item:global:1:node:1"
            left_node_item = window._find_tree_panel_item(root, left_node_key)
            self.assertIsNotNone(left_node_item)
            window._select_preview_payload({"nodes": ["1"]})
            QApplication.processEvents()
            synced_boundary_key = str(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole))
            self.assertTrue(synced_boundary_key.startswith("boundary_condition:item:"))
            self.assertTrue(synced_boundary_key.endswith(":node:1"))
            window.delete_left_tree_boundary_condition_item(left_node_key)
            QApplication.processEvents()
            self.assertEqual(window.cfg["boundary_conditions"][0]["nodes"], ["2"])

            root = window.tree.topLevelItem(0)
            fixed_type_item = window._find_tree_panel_item(root, "boundary_condition:type:fixed")
            self.assertIsNotNone(fixed_type_item)
            window.tree.setCurrentItem(fixed_type_item)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "boundary_conditions")
            self.assertEqual(window._active_boundary_condition_kind, "fixed")
            self.assertIn("1", window._selected_preview_entities()["nodes"])
            window._select_preview_payload({"nodes": ["2"]})
            before_boundary_count = len(window.cfg["boundary_conditions"])
            window.apply_active_boundary_condition_to_selected()
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["boundary_conditions"]), before_boundary_count + 1)
            self.assertEqual(window.cfg["boundary_conditions"][-1]["nodes"], ["2"])
            self.assertEqual(window.cfg["boundary_conditions"][-1]["support_type"], "fixed")
            window._select_preview_payload({"nodes": ["2"]})
            window.remove_boundary_conditions_from_selected_nodes("fixed")
            QApplication.processEvents()
            fixed_conditions = [
                raw
                for raw in window.cfg["boundary_conditions"]
                if isinstance(raw, dict) and window._boundary_condition_kind_from_spec(raw) == "fixed"
            ]
            self.assertTrue(
                all("2" not in {str(node) for node in window._ensure_list(raw.get("nodes", raw.get("node", [])))} for raw in fixed_conditions)
            )

            window._select_preview_payload({"nodes": ["3", "4"]})
            window.boundary_batch_scope.setCurrentIndex(window.boundary_batch_scope.findData("stage"))
            window.boundary_batch_kind.setCurrentIndex(window.boundary_batch_kind.findData("prescribed"))
            window.boundary_batch_ux.setText("0.015")
            window.boundary_batch_uy.setText("")
            window.add_selected_boundary_condition()
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["boundary_conditions"][0]["nodes"], ["3", "4"])
            self.assertAlmostEqual(stage["boundary_conditions"][0]["ux"], 0.015)
            self.assertEqual(window._selected_preview_entities()["nodes"], {"3", "4"})

            window._select_preview_payload({"nodes": ["1", "2", "3"]})
            window.boundary_mpc_master.setText("1")
            window.boundary_mpc_dof.setCurrentText("uy")
            window.boundary_mpc_method.setCurrentText("lagrange")
            window.boundary_mpc_coefficient.setText("1.0")
            window.boundary_mpc_value.setText("0.0")
            window.add_selected_mpc_constraints()
            stage = window.cfg["stages"][0]
            self.assertEqual(len(stage["mpc_constraints"]), 2)
            self.assertEqual({mpc["slave"] for mpc in stage["mpc_constraints"]}, {"2", "3"})
            self.assertTrue(all(mpc["method"] == "lagrange" for mpc in stage["mpc_constraints"]))
            self.assertEqual(window._selected_preview_entities()["nodes"], {"1", "2", "3"})

            window._select_preview_payload({"elements": ["1"]})
            window.boundary_hydro_kind.setCurrentIndex(window.boundary_hydro_kind.findData("flux"))
            window.boundary_hydro_value.setText("-0.25")
            window.add_selected_hydro_boundary_condition()
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["type"], "consolidation")
            self.assertIn("p", window.cfg["analysis"]["fields"])
            self.assertEqual(len(stage["hydro"]["pore_flux_bcs"][0]["edges"]), 4)
            self.assertAlmostEqual(stage["hydro"]["pore_flux_bcs"][0]["flux"], -0.25)
            self.assertEqual(window._selected_preview_entities()["elements"], {"1"})
            hydro_edge_key = next(
                key
                for key in window._boundary_navigation_specs_cache
                if key.startswith("boundary_condition:item:hydro:") and ":edge:" in key
            )
            window.delete_left_tree_boundary_condition_item(hydro_edge_key)
            QApplication.processEvents()
            self.assertEqual(len(window.cfg["stages"][0]["hydro"]["pore_flux_bcs"][0]["edges"]), 3)
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_load_gui_cases_selection_earthquake_and_seepage_pressure(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import QPointF, Qt
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtWidgets import QApplication
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u"]},
                "mesh": {
                    "nodes": {
                        "1": [0.0, 0.0],
                        "2": [1.0, 0.0],
                        "3": [1.0, 1.0],
                        "4": [0.0, 1.0],
                    },
                    "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                },
                "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
                "stages": [{"name": "LoadStage", "type": "static"}],
                "load_cases": [{"name": "LC1", "type": "static", "scale": 1.0, "active": True}],
            }
            window.populate_forms()
            window.update_preview()
            window.stage_table.selectRow(0)
            window._select_tree_panel("loads")
            window.activate_load_selection_tool()
            window.show()
            QApplication.processEvents()

            def real_click_model_item(kind: str, ident: str) -> None:
                target = next(
                    item
                    for item in window.scene.items()
                    if isinstance(item.data(0), dict)
                    and item.data(0).get("kind") == kind
                    and str(item.data(0).get("id")) == ident
                )
                pos = window.view.mapFromScene(target.sceneBoundingRect().center())
                press = QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress,
                    QPointF(pos),
                    QPointF(window.view.viewport().mapToGlobal(pos)),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                release = QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease,
                    QPointF(pos),
                    QPointF(window.view.viewport().mapToGlobal(pos)),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                window.view.mousePressEvent(press)
                window.view.mouseReleaseEvent(release)
                QApplication.processEvents()

            real_click_model_item("element", "1")
            self.assertEqual(window._selected_preview_entities()["elements"], {"1"})
            window.scene.clearSelection()
            edge_item = next(
                item
                for item in window.scene.items()
                if isinstance(item.data(0), dict)
                and item.data(0).get("kind") == "edge"
                and str(item.data(0).get("id")) == "1-2"
            )
            right_click_pos = window.view.mapFromScene(edge_item.sceneBoundingRect().center())
            context_calls: list[str] = []
            original_context_menu = window._show_command_context_menu
            window._show_command_context_menu = lambda _global_pos, context: context_calls.append(context)
            try:
                window.view.contextMenuEvent(
                    type(
                        "FakeLoadContextMenuEvent",
                        (),
                        {
                            "pos": lambda self, p=right_click_pos: p,
                            "globalPos": lambda self, p=window.view.viewport().mapToGlobal(right_click_pos): p,
                            "accept": lambda self: None,
                        },
                    )()
                )
            finally:
                window._show_command_context_menu = original_context_menu
            self.assertEqual(context_calls, ["view"])
            self.assertEqual(window._selected_preview_entities()["edges"], [("1", "2")])
            context_labels = {label for label, _callback, _tip in window._context_menu_action_specs("view") if label != "-"}
            self.assertIn("選択要素材料へ体積力", context_labels)
            self.assertIn("選択辺/要素へ偏分布面荷重", context_labels)
            window.scene.clearSelection()

            window.load_case_table.setRowCount(0)
            window.add_load_case_row(name="EQ1", case_type="earthquake", scale="1.0", active="true", description="pseudo static")
            window.apply_load_cases_panel()
            window.stage_table.selectRow(0)
            window.load_case_selector.setCurrentText("EQ1")
            window.load_batch_scope.setCurrentIndex(window.load_batch_scope.findData("stage"))

            window._select_preview_payload({"nodes": ["3", "4"]})
            self.assertIn("節点 2", window.load_selection_summary.text())
            window.load_fx.setText("5.5")
            window.load_fy.setText("-2.0")
            window.add_selected_nodal_load_condition()
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["loads"][0]["nodes"], ["3", "4"])
            self.assertEqual(stage["loads"][0]["load_case"], "EQ1")
            self.assertAlmostEqual(stage["loads"][0]["fx"], 5.5)
            self.assertEqual(window._selected_preview_entities()["nodes"], {"3", "4"})

            window._select_preview_payload({"elements": ["1"]})
            window.load_tx.setText("0.0")
            window.load_ty.setText("-12.0")
            window.add_selected_distributed_load_condition()
            stage = window.cfg["stages"][0]
            self.assertEqual(len(stage["loads"][1]["edges"]), 4)
            self.assertAlmostEqual(stage["loads"][1]["ty"], -12.0)
            self.assertEqual(window._selected_preview_entities()["elements"], {"1"})

            window._select_preview_payload({"elements": ["1"]})
            window.load_body_material.setCurrentIndex(window.load_body_material.findData("__selected__"))
            window.load_body_bx.setText("0.0")
            window.load_body_by.setText("-18.5")
            window.add_selected_body_load_condition()
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["loads"][2]["type"], "body")
            self.assertEqual(stage["loads"][2]["material"], "soil")
            self.assertAlmostEqual(stage["loads"][2]["by"], -18.5)
            self.assertEqual(window._selected_preview_entities()["elements"], {"1"})

            window._select_preview_payload({"edges": [["1", "2"]]})
            window.load_surface_distribution.setCurrentIndex(window.load_surface_distribution.findData("linear"))
            window.load_tx.setText("0.0")
            window.load_ty.setText("-1.0")
            window.load_tx_end.setText("0.0")
            window.load_ty_end.setText("-3.0")
            window.add_selected_distributed_load_condition(distribution="linear")
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["loads"][3]["distribution"], "linear")
            self.assertAlmostEqual(stage["loads"][3]["ty1"], -1.0)
            self.assertAlmostEqual(stage["loads"][3]["ty2"], -3.0)
            self.assertEqual(window._selected_preview_entities()["edges"], [("1", "2")])

            window.load_seismic_kh.setText("0.18")
            window.load_seismic_kv.setText("0.03")
            window.load_seismic_direction.setCurrentText("-X")
            window.add_panel_earthquake_load()
            stage = window.cfg["stages"][0]
            seismic = stage["loads"][4]
            self.assertEqual(seismic["type"], "gravity")
            self.assertLess(seismic["gx"], 0.0)
            self.assertEqual(seismic["seismic"]["method"], "pseudo_static")
            self.assertEqual(seismic["load_case"], "EQ1")

            window._refresh_tree(select_panel="loads")
            QApplication.processEvents()
            root = window.tree.topLevelItem(0)
            node_type_item = window._find_tree_panel_item(root, "load_condition:type:node")
            edge_type_item = window._find_tree_panel_item(root, "load_condition:type:edge")
            body_type_item = window._find_tree_panel_item(root, "load_condition:type:body")
            self.assertIsNotNone(node_type_item)
            self.assertIsNotNone(edge_type_item)
            self.assertIsNotNone(body_type_item)
            node_item = window._find_tree_panel_item(root, "load_condition:item:stage:1:1:node:3")
            self.assertIsNotNone(node_item)
            window._select_preview_payload({"nodes": ["3"]})
            QApplication.processEvents()
            self.assertEqual(str(window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole)), "load_condition:item:stage:1:1:node:3")
            window.tree.setCurrentItem(node_item)
            QApplication.processEvents()
            self.assertEqual(window.current_panel_key, "loads")
            self.assertEqual(window._selected_preview_entities()["nodes"], {"3"})
            window.delete_left_tree_load_condition_item("load_condition:item:stage:1:1:node:3")
            QApplication.processEvents()
            stage = window.cfg["stages"][0]
            self.assertEqual(stage["loads"][0]["nodes"], ["4"])
            edge_item = window._find_tree_panel_item(window.tree.topLevelItem(0), "load_condition:item:stage:1:2:edge:1-2")
            self.assertIsNotNone(edge_item)
            window.delete_left_tree_load_condition_item("load_condition:item:stage:1:2:edge:1-2")
            QApplication.processEvents()
            stage = window.cfg["stages"][0]
            self.assertEqual(len(stage["loads"][1]["edges"]), 3)
            self.assertTrue(all(tuple(edge) != ("1", "2") for edge in stage["loads"][1]["edges"]))
            before_add_count = len(stage["loads"])
            window._select_preview_payload({"nodes": ["1"]})
            window.add_selected_load_from_tree_payload({"kind": "node"})
            QApplication.processEvents()
            stage = window.cfg["stages"][0]
            self.assertEqual(len(stage["loads"]), before_add_count + 1)
            self.assertEqual(stage["loads"][-1]["nodes"], ["1"])

            with tempfile.TemporaryDirectory() as tmp:
                csv_path = Path(tmp) / "seepage.csv"
                csv_path.write_text("node_id,pore_pressure\n1,10.5\n3,7.25\n", encoding="utf-8")
                window.import_pore_pressure_csv_path(csv_path, selected_stage=True)

            stage = window.cfg["stages"][0]
            self.assertEqual(stage["type"], "consolidation")
            self.assertIn("p", window.cfg["analysis"]["fields"])
            self.assertEqual(len(stage["hydro"]["pressure_bcs"]), 2)
            self.assertEqual(stage["hydro"]["pressure_bcs"][0]["source"], "seepage_csv")
            self.assertEqual(stage["hydro"]["pressure_bcs"][0]["load_case"], "EQ1")
            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])

    def test_p0_selection_stage_diff_and_post_controls(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QGraphicsView, QTableWidgetItem
        except ModuleNotFoundError:
            self.skipTest("PySide6 is not installed")

        from geofem_app.gui.main_window import run_gui

        original_exec = QApplication.exec
        checks: list[dict[str, object]] = []

        def fake_exec(app: QApplication) -> int:
            windows = [widget for widget in QApplication.topLevelWidgets() if widget.__class__.__name__ == "MainWindow"]
            self.assertTrue(windows)
            window = windows[0]
            window.cfg = {
                "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
                "mesh": {
                    "nodes": {
                        "1": [0.0, 0.0],
                        "2": [1.0, 0.0],
                        "3": [1.0, 1.0],
                        "4": [0.0, 1.0],
                        "5": [2.0, 0.0],
                        "6": [2.0, 1.0],
                    },
                    "elements": [
                        {"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"},
                        {"id": "2", "type": "QUAD4", "nodes": ["2", "5", "6", "3"], "material": "clay"},
                    ],
                    "node_sets": {"left": ["1", "4"], "top": ["4", "3", "6"]},
                    "element_sets": {"all": ["1", "2"], "right": ["2"]},
                    "blocks": {"B_right": {"elements": ["2"]}},
                },
                "materials": {
                    "soil": {"model": "elastic", "E": 1000.0, "nu": 0.3},
                    "clay": {"model": "elastic", "E": 800.0, "nu": 0.35},
                    "rock": {"model": "elastic", "E": 5000.0, "nu": 0.25},
                },
                "stages": [
                    {"name": "initial", "type": "static", "boundary_conditions": [{"set": "left", "ux": 0.0}]},
                    {
                        "name": "excavation",
                        "type": "death",
                        "elements": ["2"],
                        "stress_release": 0.5,
                        "element_properties": [{"element": "1", "material": "rock"}],
                        "boundary_conditions": [{"nodes": ["1", "4"], "ux": 1.0, "uy": 0.0}],
                        "loads": [{"edge": ["2", "3"], "ty": -5.0}],
                        "hydro": {"pressure_bcs": [{"edge": ["3", "4"], "pressure": 2.0}]},
                    },
                ],
            }
            window.populate_forms()
            window.update_preview()

            window.cfg["analysis"]["unit_system"] = "m-N"
            window._after_form_change("undo target")
            self.assertEqual(window.cfg["analysis"]["unit_system"], "m-N")
            window.undo_edit()
            self.assertEqual(window.cfg["analysis"].get("unit_system", "m-kN"), "m-kN")
            window.redo_edit()
            self.assertEqual(window.cfg["analysis"]["unit_system"], "m-N")
            window.edit_undo_granularity.setCurrentIndex(window.edit_undo_granularity.findData("manual"))
            window.create_manual_edit_undo_point("manual point")
            window.cfg["analysis"]["unit_system"] = "cm-kN"
            window._after_form_change("manual granular edit")
            self.assertEqual(window.cfg["analysis"]["unit_system"], "cm-kN")
            window.undo_edit()
            self.assertEqual(window.cfg["analysis"]["unit_system"], "m-N")
            window.edit_undo_granularity.setCurrentIndex(window.edit_undo_granularity.findData("operation"))

            window.set_selection_mode("rectangle")
            self.assertEqual(window.view.dragMode(), QGraphicsView.DragMode.RubberBandDrag)
            self.assertIn("矩形選択", window.selection_help_label.text())
            window.selection_expr_field.setCurrentText("blocks")
            window.selection_expr_operator.setCurrentText("in")
            window.selection_expr_value.setText("B_right")
            self.assertEqual(window.build_selection_expression_from_gui(), "'B_right' in blocks")
            selected = window.select_by_filter(kind="material", value="clay")
            self.assertGreaterEqual(selected, 1)
            self.assertEqual(window._selected_preview_entities()["elements"], {"2"})
            window.save_named_selection("right_named")
            self.assertEqual(window.named_selection_table.rowCount(), 1)
            window.select_by_filter(kind="material", value="clay")
            window.save_current_selection_as_set("sel_p0")
            self.assertEqual(window.cfg["mesh"]["element_sets"]["sel_p0"], ["2"])
            self.assertGreaterEqual(len(window.selection_history), 2)
            window.select_by_filter(kind="material", value="clay")
            window.selection_operation.setCurrentIndex(window.selection_operation.findData("add"))
            window.select_by_filter(kind="nodes", value="1", mode=window._current_selection_operation())
            self.assertIn("1", window._selected_preview_entities()["nodes"])
            self.assertIn("2", window._selected_preview_entities()["elements"])
            window.save_named_selection("mixed_named")
            comparison = window.compare_named_selections("right_named", "mixed_named")
            self.assertEqual(comparison["elements"]["common"], ["2"])
            self.assertEqual(comparison["nodes"]["only_right"], ["1"])
            saved_comparison = window.save_selection_comparison("right_named", "mixed_named", "right_vs_mixed")
            self.assertIn("right_vs_mixed", window.cfg["selection_comparisons"])
            self.assertGreaterEqual(window.selection_compare_table.rowCount(), 1)
            self.assertEqual(saved_comparison["left"], "right_named")
            window.restore_named_selection("right_named")
            self.assertEqual(window._selected_preview_entities()["elements"], {"2"})
            window.select_by_filter(kind="block", value="B_right")
            self.assertEqual(window._selected_preview_entities()["elements"], {"2"})
            window.select_by_expression('"right" in sets or "B_right" in blocks')
            self.assertEqual(window._selected_preview_entities()["elements"], {"2"})
            current_index = window.selection_history_index
            window.select_by_filter(kind="nodes", value="4")
            self.assertEqual(window._selected_preview_entities()["nodes"], {"4"})
            window.undo_selection()
            self.assertEqual(window.selection_history_index, current_index)
            self.assertEqual(window._selected_preview_entities()["elements"], {"2"})
            window.redo_selection()
            self.assertEqual(window._selected_preview_entities()["nodes"], {"4"})
            window.restore_selection_history(current_index)
            self.assertEqual(window.selection_history_table.rowCount(), len(window.selection_history))

            window.stage_table.selectRow(1)
            window.refresh_stage_difference_table()
            self.assertGreater(window.stage_diff_table.rowCount(), 0)
            window.refresh_stage_cross_compare_table()
            self.assertEqual(window.stage_compare_table.rowCount(), 2)
            self.assertEqual(window.stage_compare_table.horizontalHeaderItem(2).text(), "要素")
            self.assertEqual(window.stage_compare_table.horizontalHeaderItem(3).text(), "材料")
            for col in (2, 3, 4, 5, 6):
                payload = window.stage_compare_table.item(1, col).data(Qt.ItemDataRole.UserRole)
                self.assertIsInstance(payload, dict)
                self.assertTrue(payload["changed"])
                self.assertEqual(window.stage_compare_table.item(1, col).background().color().name().lower(), "#fff3cd")
            window.refresh_stage_conflict_table()
            self.assertGreater(window.stage_conflict_table.rowCount(), 0)
            first_conflict = window.stage_conflict_table.item(0, 0).data(Qt.ItemDataRole.UserRole)
            self.assertIn("suggestion", first_conflict)
            window.repair_stage_conflict(first_conflict)
            self.assertLessEqual(len(window.collect_stage_cumulative_conflicts()), window.stage_conflict_table.rowCount())
            self.assertIn("death", window.stage_guidance_label.text())
            self.assertGreater(window.stage_wizard_table.rowCount(), 0)
            window.approve_stage_difference("loads", note="checked", approver="qa-user", locked=False)
            self.assertTrue(any("loads" in key for key in window.cfg["stage_diff_approvals"]))
            approval_record = next(value for key, value in window.cfg["stage_diff_approvals"].items() if "loads" in key)
            self.assertEqual(approval_record["approver"], "qa-user")
            self.assertFalse(approval_record["locked"])
            window.reject_stage_difference("loads", note="needs revision", approver="reviewer")
            window.reapprove_stage_difference("loads", note="fixed", approver="lead", locked=False)
            history_compare = window.compare_stage_approval_history("loads")
            self.assertTrue(history_compare["ok"])
            self.assertTrue(history_compare["changed_status"])
            self.assertGreaterEqual(window.stage_approval_history_table.rowCount(), 0)
            window.update_preview()
            self.assertTrue(any((item.data(0) or {}).get("kind") == "stage_diff" for item in window.scene.items()))
            window.select_by_filter(kind="stage_condition", value="")
            entities = window._selected_preview_entities()
            self.assertTrue(entities["nodes"] or entities["edges"] or entities["elements"])
            loads_row = next(row for row in range(window.stage_diff_table.rowCount()) if window.stage_diff_table.item(row, 1).text() == "loads")
            window.repair_stage_difference_cell(loads_row, 3)
            self.assertNotIn("loads", window.cfg["stages"][1])
            window.cfg["stages"][1]["loads"] = [{"edge": ["2", "3"], "ty": -5.0}]
            window.refresh_stage_difference_table()
            loads_row = next(row for row in range(window.stage_diff_table.rowCount()) if window.stage_diff_table.item(row, 1).text() == "loads")
            window.repair_stage_difference_row(loads_row)
            self.assertNotIn("loads", window.cfg["stages"][1])
            window.apply_stage_construction_template("boundary_change")
            self.assertTrue(window.cfg["stages"][1]["template"]["boundary_change"])
            with tempfile.TemporaryDirectory() as tmpdir:
                template_path = Path(tmpdir) / "stage_templates.yaml"
                self.assertEqual(window.save_stage_template_library(template_path), template_path)
                window.cfg["stage_template_library"] = {}
                window.load_stage_template_library(template_path)
                self.assertIn("掘削:応力解放", window._stage_template_library())
                self.assertIn("道路土工:段階掘削", window._stage_template_library())
                self.assertIn("Road earthwork: excavation with release", window._stage_template_library())
                self.assertIn("River levee: rapid drawdown check", window._stage_template_library())
                self.assertIn("GeoFEAS public: Tunnel excavation and stress release", window._stage_template_library())
                window.cfg["stages"][1]["geofeas_workflow"] = "retaining_excavation"
                window.refresh_stage_guidance()
                self.assertIn("GeoFEAS public workflow: retaining_excavation", window.stage_guidance_label.text())
                self.assertTrue(any("stage:" in step[1] for step in window.stage_guidance_steps(1)))

            window.result_element_values = {"1": 1.2, "2": 2.4}
            window.post_component = "q"
            window.post_mode = "contour"
            window.show_contour_labels.setChecked(True)
            window.show_contour_lines.setChecked(True)
            window.contour_interpolation.setCurrentText("曲線補間")
            window.contour_level_count.setText("5")
            window.contour_curve_segments.setText("3")
            window.show_element_boundaries.setChecked(False)
            window.smooth_contours.setChecked(True)
            window.legend_min_edit.setText("1.0")
            window.legend_max_edit.setText("2.0")
            window.clip_contours_to_legend.setChecked(True)
            window.update_preview()
            self.assertTrue(any((item.data(0) or {}).get("kind") == "contour_label" for item in window.scene.items()))
            self.assertTrue(any((item.data(0) or {}).get("kind") == "contour_line" for item in window.scene.items()))
            self.assertTrue(any((item.data(0) or {}).get("curve") for item in window.scene.items() if isinstance(item.data(0), dict)))
            self.assertFalse(any((item.data(0) or {}).get("kind") == "edge" for item in window.scene.items()))
            window.measure_line_start.setText("0,0")
            window.measure_line_end.setText("2,0.5")
            window.create_measurement_distribution()
            self.assertEqual(window.post_mode, "distribution")
            self.assertGreaterEqual(len(window.result_distribution), 1)
            self.assertIsNotNone(window._scene_image_with_layout())
            verification = window.verify_post_view_render()
            self.assertTrue(verification["ok"])
            with tempfile.TemporaryDirectory() as tmpdir:
                window.project_root = Path(tmpdir)
                snapshot = window.add_current_view_to_drawing_layout()
                self.assertIsNotNone(snapshot)
                self.assertEqual(window.drawing_layout_table.rowCount(), 1)
                self.assertEqual(len(window._drawing_layout_specs()), 1)
                drawing_template = Path(tmpdir) / "drawing_template.yaml"
                self.assertEqual(window.save_drawing_template(drawing_template), drawing_template)
                window.drawing_title_edit.setText("changed")
                window.load_drawing_template(drawing_template)
                self.assertNotEqual(window.drawing_title_edit.text(), "changed")
                self.assertIn("企業様式:照査付き", window._drawing_template_library())
                self.assertIn("Construction review A3 landscape", window._drawing_template_library())
                installed_templates = window.install_project_drawing_templates(Path(tmpdir) / "templates" / "drawing_templates.yaml")
                self.assertTrue(installed_templates.exists())
                self.assertIn("Client submission A4", window._drawing_template_library())
                baseline = Path(tmpdir) / "baseline.png"
                self.assertEqual(window.save_post_baseline(baseline), baseline)
                diff = window.compare_post_to_baseline(baseline)
                self.assertTrue(diff["ok"])
                ci = window.write_post_image_diff_ci_job(Path(tmpdir) / "post-image-diff.yml", baseline=baseline)
                self.assertTrue(ci.exists())
                self.assertIn("tools/post_image_diff_ci.py", ci.read_text(encoding="utf-8"))
                self.assertTrue((Path(tmpdir) / "post-image-diff-cases.yaml").exists())
                from geofem_app.post_image_diff import compare_images, create_sample_post_image
                from tools.post_image_diff_ci import main as post_diff_main

                generated = create_sample_post_image(Path(tmpdir) / "generated_post.png")
                self.assertTrue(compare_images(generated, generated)["ok"])
                matrix = Path(tmpdir) / "matrix.yaml"
                matrix.write_text(
                    "post_image_diff_cases:\n"
                    "  - name: contour\n"
                    f"    baseline: {Path(tmpdir).as_posix()}/matrix_baseline.png\n"
                    "    threshold: 0.02\n"
                    "    generate_sample: true\n"
                    "  - name: srm_fl\n"
                    f"    baseline: {Path(tmpdir).as_posix()}/matrix_srm.png\n"
                    "    threshold: 0.02\n"
                    "    generate_sample: true\n",
                    encoding="utf-8",
                )
                self.assertEqual(post_diff_main(["--matrix", str(matrix), "--out", str(Path(tmpdir) / "matrix_result.json")], emit=False), 0)
                window.add_report_text_block("判定", "OK")
                self.assertGreaterEqual(window.report_page_table.rowCount(), 1)
                window.report_page_table.selectRow(0)
                before_x = float(window.report_page_table.item(0, 3).text())
                window.nudge_selected_report_block(dx=0.02)
                self.assertGreater(float(window.report_page_table.item(0, 3).text()), before_x)
                window.refresh_report_canvas()
                block = next(item for item in window.report_layout_scene.items() if isinstance(item.data(0), dict) and item.data(0).get("kind") == "report_block")
                before_drag_x = float(window.report_page_table.item(0, 3).text())
                block.setPos(block.pos().x() + 20.0, block.pos().y())
                window.apply_report_canvas_positions()
                self.assertGreater(float(window.report_page_table.item(0, 3).text()), before_drag_x)
                window.refresh_report_canvas()
                handle = next(item for item in window.report_layout_scene.items() if isinstance(item.data(0), dict) and item.data(0).get("kind") == "report_resize_handle")
                before_w = float(window.report_page_table.item(0, 5).text())
                handle.setPos(handle.pos().x() + 15.0, handle.pos().y() + 8.0)
                window.apply_report_canvas_positions()
                self.assertGreater(float(window.report_page_table.item(0, 5).text()), before_w)
                html = window.build_report_wysiwyg_html()
                self.assertIn("判定", html)
                self.assertIsNotNone(window.add_current_post_to_report_page())

            window.stage_boundary_table.setRowCount(1)
            window.stage_boundary_table.setItem(0, 0, QTableWidgetItem("left"))
            window.stage_boundary_table.setItem(0, 1, QTableWidgetItem("bad-number"))
            with self.assertRaises(ValueError):
                window._float_table_cell(window.stage_boundary_table, 0, 1, "ux")
            self.assertGreater(window.cell_error_table.rowCount(), 0)
            window.jump_to_cell_error(0)
            self.assertEqual(window.stage_boundary_table.currentRow(), 0)
            window.fix_all_cell_errors()
            self.assertEqual(window.stage_boundary_table.item(0, 1).text(), "0.0")
            self.assertEqual(window.cell_error_table.rowCount(), 0)

            with tempfile.TemporaryDirectory() as tmpdir:
                window.project_root = Path(tmpdir)
                window.current_input = Path(tmpdir) / "recent.yaml"
                window.yaml_editor.setPlainText("analysis:\n  dimension: 2D\n")
                window.lock_project()
                self.assertTrue((Path(tmpdir) / ".geofem_project.lock").exists())
                autosave = window.autosave_current_project()
                self.assertIsNotNone(autosave)
                self.assertTrue(autosave.exists())
                self.assertGreaterEqual(len(window.recovery_candidates()), 2)
                recovery_compare = window.compare_recovery_candidates(0, 1)
                self.assertTrue(recovery_compare["ok"])
                window.refresh_recovery_candidates_table()
                self.assertGreaterEqual(window.recovery_candidate_table.rowCount(), 2)
                self.assertTrue(window.restore_recovery_candidate(0))
                window.unlock_project()
                self.assertFalse((Path(tmpdir) / ".geofem_project.lock").exists())
                (Path(tmpdir) / ".geofem_project.lock").write_text("user: other\nhost: other-host\n", encoding="utf-8")
                self.assertTrue(window.force_unlock_project("stale lock"))
                handoff = window.handoff_project_lock(getpass.getuser(), "shift change")
                self.assertEqual(handoff["handoff_note"], "shift change")
                window.force_unlock_project("handoff complete")
                window.refresh_audit_log_table()
                self.assertGreaterEqual(window.audit_log_table.rowCount(), 1)
                recent_file = Path(tmpdir) / "recent.yaml"
                recent_file.write_text(window.yaml_editor.toPlainText(), encoding="utf-8")
                window.record_recent_project(recent_file)
                self.assertIn(str(recent_file), window.recent_projects)

            checks.append({"ok": True})
            window.close()
            return 0

        QApplication.exec = fake_exec
        try:
            self.assertEqual(run_gui(), 0)
        finally:
            QApplication.exec = original_exec
        self.assertEqual(checks, [{"ok": True}])


class CadImportTests(unittest.TestCase):
    def test_sxf_cartesian_point_line_records_import_as_segments(self) -> None:
        text = """
ISO-10303-21;
#1=CARTESIAN_POINT('',(0.,0.));
#2=CARTESIAN_POINT('',(10.,0.));
#3=SXF_LINE_FEATURE('',#1,#2);
END-ISO-10303-21;
"""
        self.assertEqual(parse_sxf_lines(text), [(0.0, 0.0, 10.0, 0.0)])

    def test_sxf_document_preserves_layer_text_and_dimension_records(self) -> None:
        text = """
ISO-10303-21;
#1=SXF_LAYER('strata','red','dash','civil');
#2=CARTESIAN_POINT('',(0.,0.));
#3=CARTESIAN_POINT('',(10.,0.));
#4=SXF_LINE_FEATURE('base',#1,#2,#3);
#5=SXF_TEXT('ground line',#1,#2);
#6=SXF_DIMENSION('10m',#1,#2,#3);
#7=CARTESIAN_POINT('',(10.,1.));
#8=SXF_HATCH('ANSI31',#1,#2,#3,#7);
#9=SXF_LINETYPE('chain','dash dot',0.8,-0.2,0.,-0.2);
#10=SXF_LINE_FEATURE('styled',#1,#9,#2,#3);
END-ISO-10303-21;
"""
        doc = parse_sxf_document(text)
        self.assertEqual(doc["linetypes"][0]["name"], "chain")
        self.assertEqual(doc["linetypes"][0]["pattern"], [0.8, -0.2, 0.0, -0.2])
        self.assertEqual(doc["layers"][0]["name"], "strata")
        self.assertEqual(doc["layers"][0]["color"], "red")
        self.assertEqual(doc["layers"][0]["linetype"], "dash")
        self.assertEqual(doc["layers"][0]["parent"], "civil")
        self.assertEqual(doc["lines"][0]["layer"], "strata")
        self.assertEqual(doc["lines"][1]["linetype"], "chain")
        self.assertEqual(doc["annotations"][0]["text"], "ground line")
        self.assertEqual(doc["annotations"][0]["layer"], "strata")
        self.assertEqual(doc["dimensions"][0]["text"], "10m")
        self.assertAlmostEqual(doc["dimensions"][0]["value"], 10.0)
        self.assertEqual(doc["hatches"][0]["pattern"], "ANSI31")
        self.assertEqual(doc["hatches"][0]["layer"], "strata")

    def test_sxf_curve_records_preserve_curves_and_closed_circle_segments(self) -> None:
        text = """
ISO-10303-21;
#1=SXF_LAYER('curves','blue','solid','civil');
#2=SXF_ARC_FEATURE('arc',#1,(0.,0.),1.,0.,90.);
#3=SXF_CIRCLE_FEATURE('circle',#1,(2.,0.),0.5);
#4=CARTESIAN_POINT('',(0.,0.));
#5=CARTESIAN_POINT('',(1.,0.5));
#6=CARTESIAN_POINT('',(2.,0.));
#7=SXF_BEZIER_FEATURE('spline',#1,#4,#5,#6);
END-ISO-10303-21;
"""
        doc = parse_sxf_document(text)
        self.assertEqual(len(doc["curves"]), 3)
        self.assertEqual(doc["curves"][0]["type"], "arc")
        self.assertEqual(doc["curves"][1]["type"], "circle")
        self.assertTrue(doc["curves"][1]["closed"])
        self.assertEqual(doc["curves"][1]["segment_count"], 30)
        self.assertEqual(doc["curves"][2]["type"], "bezier")
        self.assertEqual(len(doc["curves"][2]["points"]), 25)
        self.assertEqual(len(parse_sxf_lines(text)), sum(curve["segment_count"] for curve in doc["curves"]))

    def test_sxf_document_preserves_all_step_entities_and_coverage(self) -> None:
        text = """
ISO-10303-21;
#1=SXF_LAYER('site','green','solid','civil');
#2=CARTESIAN_POINT('',(0.,0.));
#3=CARTESIAN_POINT('',(2.,0.));
#4=CARTESIAN_POINT('',(2.,1.));
#5=CARTESIAN_POINT('',(0.,1.));
#6=SXF_POLYGON_FEATURE('ground',#1,#2,#3,#4,#5);
#7=SXF_SYMBOL_FEATURE('borehole',#1,#2);
#8=SXF_IMAGE_FEATURE('photo',#1,(0.,0.),(1.,1.),'photo.png');
#9=SXF_UNKNOWN_FEATURE('raw payload',#2,42.);
END-ISO-10303-21;
"""
        doc = parse_sxf_document(text)
        by_id = {entity["id"]: entity for entity in doc["entities"]}
        self.assertEqual(len(by_id), 9)
        self.assertEqual(doc["sxf_entity_coverage"]["total"], 9)
        self.assertEqual(doc["sxf_entity_coverage"]["unhandled"], 1)
        self.assertIn("SXF_UNKNOWN_FEATURE", doc["sxf_entity_coverage"]["unhandled_types"])
        self.assertEqual(by_id["#6"]["mapped_as"], ["region"])
        self.assertEqual(by_id["#7"]["mapped_as"], ["metadata"])
        self.assertFalse(by_id["#9"]["handled"])
        self.assertEqual(by_id["#9"]["refs"], ["#2"])
        self.assertEqual(doc["regions"][0]["layer"], "site")
        self.assertEqual(len(parse_sxf_lines(text)), 4)

    def test_sxf_unknown_only_file_keeps_raw_entities_without_plain_fallback(self) -> None:
        text = """
ISO-10303-21;
#1=SXF_CUSTOM_PRODUCT_FEATURE('custom',#99,12.5);
#2=SXF_USER_DEFINED_ATTRIBUTE('owner','GeoFEM');
END-ISO-10303-21;
"""
        doc = parse_sxf_document(text)
        self.assertEqual(doc["format"], "sxf")
        self.assertEqual([entity["type"] for entity in doc["entities"]], ["SXF_CUSTOM_PRODUCT_FEATURE", "SXF_USER_DEFINED_ATTRIBUTE"])
        self.assertEqual(doc["sxf_entity_coverage"]["total"], 2)
        self.assertEqual(doc["sxf_entity_coverage"]["unhandled"], 1)
        self.assertEqual(doc["sxf_entity_coverage"]["handled"], 1)
        self.assertEqual(parse_sxf_lines(text), [])

    def test_complex_p21_roundtrip_preserves_multiline_raw_attributes(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "cad" / "complex_site.p21"
        text = fixture.read_text(encoding="utf-8")
        records = split_sxf_step_records(text)
        self.assertTrue(any("base;with semicolon" in record for record in records))
        doc = parse_sxf_document(text)
        self.assertEqual(doc["sxf_entity_coverage"]["total"], 10)
        self.assertEqual(doc["lines"][0]["layer"], "site layer")
        self.assertEqual(doc["lines"][0]["linetype"], "chain")
        self.assertEqual(doc["linetypes"][0]["pattern"], [0.8, -0.2, 0.0, -0.2])
        exported = export_sxf_document(doc)
        roundtrip = parse_sxf_document(exported)
        self.assertEqual(roundtrip["sxf_entity_coverage"]["total"], 10)
        self.assertEqual(roundtrip["lines"][0]["layer"], "site layer")
        self.assertEqual(roundtrip["lines"][0]["linetype"], "chain")
        self.assertTrue(any(entity["raw"].count("\n") >= 2 for entity in roundtrip["entities"] if entity["id"] == "#3"))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / fixture.name
            out = Path(tmp) / "complex_reexport.p21"
            source.write_text(text, encoding="utf-8")
            report = validate_sxf_roundtrip(source, out)
            self.assertTrue(report["ok"], report)
            self.assertTrue(out.exists())
            self.assertEqual(report["line_mismatch_count"], 0)
            self.assertEqual(report["attributes"]["missing_entity_raw_count"], 0)

    def test_generated_sxf_export_keeps_geometry_attributes_without_raw_entities(self) -> None:
        doc = {
            "format": "gf1",
            "layers": [{"name": "soil", "color": "brown", "linetype": "dash"}],
            "linetypes": [{"name": "dash", "pattern": [1.0, -0.5]}],
            "lines": [{"id": "L1", "layer": "soil", "linetype": "dash", "start": [0.0, 0.0], "end": [2.0, 0.0]}],
            "annotations": [{"id": "T1", "layer": "soil", "point": [0.0, 0.0], "text": "bench"}],
            "dimensions": [{"id": "D1", "layer": "soil", "start": [0.0, 0.0], "end": [2.0, 0.0], "text": "2m"}],
        }
        exported = export_sxf_document(doc, preserve_raw=False)
        roundtrip = parse_sxf_document(exported)
        self.assertEqual(parse_sxf_lines(exported), [(0.0, 0.0, 2.0, 0.0)])
        self.assertEqual(roundtrip["layers"][0]["name"], "soil")
        self.assertEqual(roundtrip["lines"][0]["layer"], "soil")
        self.assertEqual(roundtrip["lines"][0]["linetype"], "dash")
        self.assertEqual(roundtrip["annotations"][0]["text"], "bench")
        self.assertEqual(roundtrip["dimensions"][0]["text"], "2m")

    def test_dxf_document_preserves_layer_color_linetype_text_and_dimension(self) -> None:
        dxf = """0
SECTION
2
TABLES
0
TABLE
2
LTYPE
0
LTYPE
2
GEO_DASH
3
Dash dot
73
4
40
1.0
49
0.5
49
-0.25
49
0
49
-0.25
0
ENDTAB
0
TABLE
2
DIMSTYLE
0
DIMSTYLE
2
geo_dim
140
2.5
41
1.25
0
ENDTAB
0
TABLE
2
LAYER
0
LAYER
2
strata
62
1
6
GEO_DASH
0
ENDTAB
0
ENDSEC
0
SECTION
2
ENTITIES
0
LINE
8
strata
10
0
20
0
11
1
21
0
0
TEXT
8
strata
10
0.5
20
0.2
1
GL
0
DIMENSION
8
strata
3
geo_dim
13
0
23
0
14
1
24
0
1
1m
0
HATCH
8
strata
2
ANSI31
70
0
92
7
10
0
20
0
10
1
20
0
10
1
20
1
92
7
10
0.2
20
0.2
10
0.4
20
0.2
10
0.4
20
0.4
0
ENDSEC
0
EOF
"""
        doc = parse_dxf_document(dxf)
        self.assertEqual(doc["linetypes"][0]["name"], "GEO_DASH")
        self.assertEqual(doc["linetypes"][0]["pattern"], [0.5, -0.25, 0.0, -0.25])
        self.assertEqual(doc["layers"][0]["name"], "strata")
        self.assertEqual(doc["layers"][0]["color"], "red")
        self.assertEqual(doc["layers"][0]["linetype"], "GEO_DASH")
        self.assertEqual(doc["lines"][0]["layer"], "strata")
        self.assertEqual(doc["annotations"][0]["text"], "GL")
        self.assertEqual(doc["dimensions"][0]["text"], "1m")
        self.assertEqual(doc["dimensions"][0]["dimension_style"], "geo_dim")
        self.assertEqual(doc["dimension_styles"][0]["name"], "geo_dim")
        self.assertAlmostEqual(doc["dimension_styles"][0]["text_height"], 2.5)
        self.assertEqual(doc["hatches"][0]["pattern"], "ANSI31")
        self.assertEqual(len(doc["hatches"][0]["rings"]), 2)
        self.assertEqual(doc["hatches"][0]["island_count"], 1)
        self.assertEqual(parse_cad_document(dxf, ".dxf")["lines"][0]["layer"], "strata")

    def test_gf1_json_geometry_imports_lines_and_regions(self) -> None:
        text = '{"geometry":{"lines":[{"start":[0,0],"end":[1,0]}],"regions":[{"points":[[0,0],[1,0],[1,1],[0,1]]}]}}'
        lines = parse_gf1_lines(text)
        self.assertIn((0.0, 0.0, 1.0, 0.0), lines)
        self.assertEqual(len(lines), 5)

    def test_gf1_region_preserves_polygon_holes(self) -> None:
        text = '{"geometry":{"regions":[{"points":[[0,0],[3,0],[3,3],[0,3]],"holes":[[[1,1],[2,1],[2,2],[1,2]]]}]}}'
        doc = parse_gf1_document(text)
        self.assertEqual(len(doc["regions"][0]["holes"]), 1)
        self.assertEqual(doc["regions"][0]["holes"][0][0], [1.0, 1.0])

    def test_gf1_document_preserves_layers_annotations_dimensions_and_tunnels(self) -> None:
        text = (
            '{"geometry":{'
            '"linetypes":[{"name":"chain","pattern":[0.8,-0.2,0,-0.2],"description":"chain"}],'
            '"layers":[{"id":"L1","name":"strata","color":"#ff0000","linetype":"dash","parent":"civil"}],'
            '"lines":[{"start":[0,0],"end":[1,0],"layer":"strata","color":"green","linetype":"chain","linetype_pattern":"0.5,-0.1,0,-0.1"}],'
            '"annotations":[{"point":[0.5,0.1],"text":"GL","layer":"notes"}],'
            '"dimensions":[{"start":[0,0],"end":[1,0],"text":"1m","layer":"dim","dimension_style":"geo_dim"}],'
            '"dimension_styles":[{"name":"geo_dim","text_height":2.5,"arrow_size":1.0}],'
            '"tunnels":[{"center":[2,2],"radius":0.5,"layer":"excavation"}],'
            '"hatches":[{"rings":[[[0,0],[2,0],[2,2],[0,2]],[[0.5,0.5],[1,0.5],[1,1]]],"layer":"strata","pattern":"ANSI31","solid":false}]'
            '}}'
        )
        doc = parse_gf1_document(text)
        self.assertEqual(doc["linetypes"][0]["name"], "chain")
        self.assertEqual(doc["linetypes"][0]["pattern"], [0.8, -0.2, 0.0, -0.2])
        self.assertEqual(doc["layers"][0]["name"], "strata")
        self.assertEqual(doc["layers"][0]["color"], "#ff0000")
        self.assertEqual(doc["layers"][0]["linetype"], "dash")
        self.assertEqual(doc["layers"][0]["parent"], "civil")
        self.assertEqual(doc["lines"][0]["layer"], "strata")
        self.assertEqual(doc["lines"][0]["color"], "green")
        self.assertEqual(doc["lines"][0]["linetype"], "chain")
        self.assertEqual(doc["lines"][0]["linetype_pattern"], [0.5, -0.1, 0.0, -0.1])
        self.assertEqual(doc["annotations"][0]["text"], "GL")
        self.assertEqual(doc["dimensions"][0]["layer"], "dim")
        self.assertEqual(doc["dimensions"][0]["dimension_style"], "geo_dim")
        self.assertEqual(doc["dimension_styles"][0]["text_height"], 2.5)
        self.assertEqual(doc["tunnels"][0]["radius"], 0.5)
        self.assertEqual(doc["hatches"][0]["pattern"], "ANSI31")
        self.assertEqual(doc["hatches"][0]["island_count"], 1)

    def test_gf1_curve_boundaries_flatten_to_region_holes_and_lines(self) -> None:
        text = (
            '{"geometry":{'
            '"regions":[{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,3]},'
            '{"type":"bezier","points":[[4,3],[3,3.5],[1,3.5],[0,3]],"segments":6},'
            '{"type":"line","start":[0,3],"end":[0,0]}'
            '],"holes":[{"type":"circle","center":[2,1.5],"radius":0.35,"segments":24}]}],'
            '"curves":[{"id":"arc1","type":"arc","center":[0,0],"radius":1,"start_angle":0,"end_angle":90,"segments":4}]'
            '}}'
        )
        doc = parse_gf1_document(text)
        self.assertEqual(len(doc["regions"]), 1)
        self.assertEqual(len(doc["regions"][0]["points"]), 9)
        self.assertEqual(len(doc["regions"][0]["holes"][0]), 24)
        self.assertEqual(len(doc["regions"][0]["curve_boundary"]), 4)
        self.assertEqual(doc["regions"][0]["curve_boundary"][2]["type"], "bezier")
        self.assertEqual(doc["regions"][0]["curve_holes"][0][0]["type"], "circle")
        self.assertAlmostEqual(doc["regions"][0]["curve_holes"][0][0]["radius"], 0.35)
        self.assertEqual(len(doc["curves"]), 1)
        self.assertEqual(doc["curves"][0]["segment_count"], 4)
        self.assertEqual(doc["curves"][0]["parameters"]["type"], "arc")
        self.assertAlmostEqual(doc["curves"][0]["parameters"]["radius"], 1.0)
        self.assertEqual(len(parse_gf1_lines(text)), 37)

    def test_gf1_nurbs_curve_boundary_preserves_parameters(self) -> None:
        text = (
            '{"geometry":{"regions":[{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,2]},'
            '{"type":"nurbs","control_points":[[4,2],[3,2.8],[1,2.8],[0,2]],"weights":[1,0.8,0.8,1],"knots":[0,0,0,0,1,1,1,1],"degree":3,"segments":8},'
            '{"type":"line","start":[0,2],"end":[0,0]}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        boundary = doc["regions"][0]["curve_boundary"]
        self.assertEqual(boundary[2]["type"], "nurbs")
        self.assertEqual(boundary[2]["degree"], 3)
        self.assertEqual(boundary[2]["weights"], [1.0, 0.8, 0.8, 1.0])
        self.assertEqual(len(boundary[2]["knots"]), 8)
        self.assertGreater(len(doc["regions"][0]["points"]), 8)

    def test_analytic_curve_boolean_graph_splits_at_direct_intersections(self) -> None:
        line = {"type": "line", "start": [-1.0, 0.0], "end": [1.0, 0.0]}
        circle = {"type": "circle", "center": [0.0, 0.0], "radius": 1.0, "closed": True}
        hits = intersect_curves(line, circle, target=0.1, tol=1.0e-9)
        self.assertEqual(len(hits), 2)
        self.assertEqual({round(hit["t_a"], 8) for hit in hits}, {0.0, 1.0})
        graph = build_analytic_curve_boolean_graph([[line], [circle]], target=0.1, tol=1.0e-9)
        self.assertEqual(graph["intersection_count"], 2)
        self.assertGreaterEqual(graph["split_edge_count"], 3)
        self.assertGreaterEqual(graph["vertex_count"], 2)

    def test_analytic_line_overlap_returns_trim_endpoints(self) -> None:
        line_a = {"type": "line", "start": [0.0, 0.0], "end": [2.0, 0.0]}
        line_b = {"type": "line", "start": [1.0, 0.0], "end": [3.0, 0.0]}
        hits = intersect_curves(line_a, line_b, target=0.1, tol=1.0e-9)
        self.assertEqual(len(hits), 2)
        self.assertEqual({round(hit["t_a"], 8) for hit in hits}, {0.5, 1.0})
        self.assertEqual({hit["method"] for hit in hits}, {"line_line_overlap"})
        graph = build_analytic_curve_boolean_graph([[line_a], [line_b]], target=0.1, tol=1.0e-9)
        self.assertEqual(graph["intersection_count"], 0)
        self.assertEqual(graph["overlap_span_count"], 1)
        self.assertEqual(graph["overlap_edge_pair_count"], 1)
        self.assertEqual(graph["overlap_edge_count"], 2)
        pair = graph["overlap_edge_pairs"][0]
        edge_a = next(edge for edge in graph["edges"] if edge["id"] == pair["edge_a"])
        edge_b = next(edge for edge in graph["edges"] if edge["id"] == pair["edge_b"])
        self.assertIn(edge_b["id"], edge_a["coincident_edge_ids"])
        self.assertIn(edge_a["id"], edge_b["coincident_edge_ids"])

    def test_analytic_curve_boolean_graph_refines_bezier_line_intersection(self) -> None:
        bezier = {"type": "bezier", "control_points": [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]}
        line = {"type": "line", "start": [0.0, 0.5], "end": [1.0, 0.5]}
        hits = intersect_curves(bezier, line, target=0.05, tol=1.0e-9)
        self.assertEqual(len(hits), 1)
        self.assertTrue(all(hit["residual"] < 1.0e-7 for hit in hits))
        self.assertAlmostEqual(hits[0]["t_a"], 0.5, places=4)
        self.assertAlmostEqual(hits[0]["t_b"], 0.5, places=4)

    def test_analytic_graph_winding_selects_union_intersection_difference_loops(self) -> None:
        region_1 = [
            {"type": "line", "start": [0.0, 0.0], "end": [3.0, 0.0]},
            {"type": "line", "start": [3.0, 0.0], "end": [3.0, 2.0]},
            {"type": "line", "start": [3.0, 2.0], "end": [0.0, 2.0]},
            {"type": "line", "start": [0.0, 2.0], "end": [0.0, 0.0]},
        ]
        region_2 = [
            {"type": "line", "start": [2.0, 0.5], "end": [5.0, 0.5]},
            {"type": "line", "start": [5.0, 0.5], "end": [5.0, 2.5]},
            {"type": "line", "start": [5.0, 2.5], "end": [2.0, 2.5]},
            {"type": "line", "start": [2.0, 2.5], "end": [2.0, 0.5]},
        ]
        graph = classify_graph_boolean_operations(build_analytic_curve_boolean_graph([region_1, region_2], target=0.25, tol=1.0e-9))
        operations = graph["boolean_operations"]
        self.assertEqual(graph["selection_engine"], "analytic_winding_containment")
        self.assertGreater(operations["union"]["edge_count"], 0)
        self.assertGreater(operations["intersection"]["edge_count"], 0)
        self.assertGreater(operations["difference_1"]["edge_count"], 0)
        self.assertGreater(operations["difference_2"]["edge_count"], 0)
        self.assertGreater(len(operations["union"]["loops"]), 0)
        self.assertTrue(any(edge.get("graph_union_interior_side") in {"left", "right"} for edge in graph["edges"]))

    def test_analytic_graph_boolean_expression_outputs_mesh_boundary_loops(self) -> None:
        region_1 = [
            {"type": "line", "start": [0.0, 0.0], "end": [3.0, 0.0]},
            {"type": "line", "start": [3.0, 0.0], "end": [3.0, 2.0]},
            {"type": "line", "start": [3.0, 2.0], "end": [0.0, 2.0]},
            {"type": "line", "start": [0.0, 2.0], "end": [0.0, 0.0]},
        ]
        region_2 = [
            {"type": "line", "start": [2.0, 0.5], "end": [5.0, 0.5]},
            {"type": "line", "start": [5.0, 0.5], "end": [5.0, 2.5]},
            {"type": "line", "start": [5.0, 2.5], "end": [2.0, 2.5]},
            {"type": "line", "start": [2.0, 2.5], "end": [2.0, 0.5]},
        ]
        self.assertEqual(parse_boolean_expression("A-B", 2)["normalized"], "(R1-R2)")
        graph = classify_graph_boolean_operations(build_analytic_curve_boolean_graph([region_1, region_2], target=0.25, tol=1.0e-9), expression="A-B")
        rings = operation_loop_polygons(graph, operation="expression", target=0.25)
        self.assertGreater(graph["boolean_operations"]["expression"]["edge_count"], 0)
        self.assertEqual(graph["boolean_expression"]["normalized"], "(R1-R2)")
        self.assertGreater(len(rings), 0)
        self.assertAlmostEqual(sum(ring["area"] for ring in rings), 4.5, places=6)
        self.assertEqual(regions_containing_point(graph, (1.0, 1.0)), [1])
        self.assertEqual(regions_containing_point(graph, (2.5, 1.0)), [1, 2])

    def test_analytic_graph_reports_tolerance_diagnostics(self) -> None:
        line = {"type": "line", "start": [0.0, 0.0], "end": [1.0, 0.0]}
        duplicate = {"type": "line", "start": [1.0, 0.0], "end": [0.0, 0.0]}
        near_gap = {"type": "line", "start": [1.000002, 0.0], "end": [2.0, 0.0]}
        graph = build_analytic_curve_boolean_graph([[line], [duplicate], [near_gap]], target=1.0, tol=1.0e-9)
        diagnostics = graph["tolerance_diagnostics"]
        self.assertGreaterEqual(diagnostics["duplicate_curve_pair_count"], 1)
        self.assertGreaterEqual(diagnostics["overlapping_curve_pair_count"], 1)
        self.assertGreaterEqual(diagnostics["tiny_gap_count"], 1)

    def test_analytic_graph_prunes_far_curve_diagnostic_pairs(self) -> None:
        line = {"type": "line", "start": [0.0, 0.0], "end": [1.0, 0.0]}
        duplicate = {"type": "line", "start": [1.0, 0.0], "end": [0.0, 0.0]}
        near_gap = {"type": "line", "start": [1.000002, 0.0], "end": [2.0, 0.0]}
        far_regions = [
            [{"type": "line", "start": [10.0 + index * 5.0, 10.0], "end": [11.0 + index * 5.0, 10.0]}]
            for index in range(24)
        ]
        graph = build_analytic_curve_boolean_graph([[line], [duplicate], [near_gap], *far_regions], target=1.0, tol=1.0e-9)
        diagnostics = graph["tolerance_diagnostics"]
        total_pairs = graph["curve_count"] * (graph["curve_count"] - 1) // 2
        self.assertLess(diagnostics["curve_pair_candidate_count"], total_pairs // 4)
        self.assertGreaterEqual(diagnostics["duplicate_curve_pair_count"], 1)
        self.assertGreaterEqual(diagnostics["overlapping_curve_pair_count"], 1)
        self.assertGreaterEqual(diagnostics["tiny_gap_count"], 1)

    def test_analytic_graph_reports_near_tangent_nurbs_diagnostics(self) -> None:
        nurbs_a = {
            "type": "nurbs",
            "control_points": [[0.0, 0.0], [0.7, 1.0], [1.4, -1.0], [2.0, 0.0]],
            "weights": [1.0, 0.85, 0.85, 1.0],
            "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "degree": 3,
        }
        nurbs_b = {
            "type": "nurbs",
            "control_points": [[0.0, 1.0e-7], [0.7, 1.0000001], [1.4, -0.9999999], [2.0, 1.0e-7]],
            "weights": [1.0, 0.85, 0.85, 1.0],
            "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "degree": 3,
        }
        graph = build_analytic_curve_boolean_graph([[nurbs_a], [nurbs_b]], target=0.1, tol=1.0e-9)
        diagnostics = graph["tolerance_diagnostics"]
        self.assertGreaterEqual(diagnostics["overlapping_curve_pair_count"], 1)
        self.assertGreaterEqual(diagnostics["duplicate_curve_pair_count"], 1)
        self.assertEqual(graph["overlap_span_count"], 1)
        self.assertEqual(graph["overlap_edge_pair_count"], 1)
        overlap_edges = [edge for edge in graph["edges"] if edge.get("overlap_span_ids")]
        self.assertEqual(len(overlap_edges), 2)
        self.assertTrue(all(edge.get("overlap_role") == "coincident_trim_edge" for edge in overlap_edges))
        self.assertTrue(all(edge.get("coincident_edge_ids") for edge in overlap_edges))

    def test_gf1_binary_payload_extracts_embedded_json_geometry(self) -> None:
        payload = b"\x00GF1BIN\x00" + b'{"geometry":{"lines":[{"start":[0,0],"end":[2,0]}]}}' + b"\x00tail"
        doc = parse_gf1_document_bytes(payload)
        self.assertTrue(doc["binary_payload"])
        self.assertEqual(doc["lines"][0]["end"], [2.0, 0.0])
        self.assertEqual(parse_gf1_lines_bytes(payload), [(0.0, 0.0, 2.0, 0.0)])

    def test_gf1_binary_payload_extracts_zlib_compressed_json_geometry(self) -> None:
        json_payload = b'{"geometry":{"regions":[{"points":[[0,0],[2,0],[2,1],[0,1]]}]}}'
        payload = b"GF1Z" + zlib.compress(json_payload) + b"\x00footer"
        doc = parse_gf1_document_bytes(payload)
        self.assertTrue(doc["binary_payload"])
        self.assertEqual(len(doc["regions"]), 1)
        self.assertEqual(len(parse_gf1_lines_bytes(payload)), 4)

    def test_gf1_binary_payload_extracts_utf16_json_geometry(self) -> None:
        json_payload = '{"geometry":{"lines":[{"start":[0,1],"end":[2,1]}]}}'.encode("utf-16-le")
        payload = b"\x00GF1UTF16\x00" + json_payload
        doc = parse_gf1_document_bytes(payload)
        self.assertTrue(doc["binary_payload"])
        self.assertEqual(doc["lines"][0]["start"], [0.0, 1.0])
        self.assertEqual(parse_gf1_lines_bytes(payload), [(0.0, 1.0, 2.0, 1.0)])

    def test_gf1_file_document_reads_binary_payload_before_text_fallback(self) -> None:
        json_payload = b'{"geometry":{"lines":[{"start":[1,0],"end":[3,0]}],"layers":[{"name":"soil"}]}}'
        with tempfile.TemporaryDirectory() as tmp:
            gf1 = Path(tmp) / "model.gf1"
            gf1.write_bytes(b"GF1Z" + zlib.compress(json_payload))
            doc = parse_cad_file_document(gf1)
            self.assertEqual(doc["source"], str(gf1))
            self.assertEqual(doc["layers"][0]["name"], "soil")
            self.assertEqual(parse_cad_file(gf1), [(1.0, 0.0, 3.0, 0.0)])

    def test_dwg_reports_converter_requirement(self) -> None:
        with self.assertRaises(CadImportError):
            parse_cad_lines("binary", ".dwg")

    def test_dwg_external_converter_output_imports_as_dxf(self) -> None:
        dxf = """0
SECTION
2
ENTITIES
0
LINE
10
0
20
0
11
1
21
0
0
ENDSEC
0
EOF
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwg = root / "model.dwg"
            dwg.write_bytes(b"fake-dwg")
            converter = root / "convert.py"
            converter.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                f"Path(sys.argv[2]).write_text({dxf!r}, encoding='utf-8')\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_cad_file(dwg, converter=[sys.executable, str(converter)]), [(0.0, 0.0, 1.0, 0.0)])
            report = validate_dwg_converter_link(dwg, converter=[sys.executable, str(converter)])
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["converted_suffix"], ".dxf")
            self.assertEqual(report["line_count"], 1)

    def test_dwg_converter_autodiscovery_uses_path_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwg = root / "model.dwg"
            dwg.write_bytes(b"fake-dwg")
            converter = root / "fake_dwg2dxf.cmd"
            converter.write_text(
                "@echo off\n"
                "> \"%~2\" echo 0\n"
                ">> \"%~2\" echo SECTION\n"
                ">> \"%~2\" echo 2\n"
                ">> \"%~2\" echo ENTITIES\n"
                ">> \"%~2\" echo 0\n"
                ">> \"%~2\" echo LINE\n"
                ">> \"%~2\" echo 10\n"
                ">> \"%~2\" echo 0\n"
                ">> \"%~2\" echo 20\n"
                ">> \"%~2\" echo 0\n"
                ">> \"%~2\" echo 11\n"
                ">> \"%~2\" echo 2\n"
                ">> \"%~2\" echo 21\n"
                ">> \"%~2\" echo 0\n"
                ">> \"%~2\" echo 0\n"
                ">> \"%~2\" echo ENDSEC\n"
                ">> \"%~2\" echo 0\n"
                ">> \"%~2\" echo EOF\n",
                encoding="utf-8",
            )
            old_path = os.environ.get("PATH", "")
            old_converter = os.environ.pop("GEOFEM_DWG_CONVERTER", None)
            old_candidates = os.environ.get("GEOFEM_DWG_CONVERTER_CANDIDATES")
            old_auto = os.environ.pop("GEOFEM_DWG_AUTODISCOVER", None)
            try:
                os.environ["PATH"] = f"{root}{os.pathsep}{old_path}"
                os.environ["GEOFEM_DWG_CONVERTER_CANDIDATES"] = converter.name
                self.assertEqual(parse_cad_file(dwg), [(0.0, 0.0, 2.0, 0.0)])
            finally:
                os.environ["PATH"] = old_path
                if old_converter is not None:
                    os.environ["GEOFEM_DWG_CONVERTER"] = old_converter
                else:
                    os.environ.pop("GEOFEM_DWG_CONVERTER", None)
                if old_candidates is not None:
                    os.environ["GEOFEM_DWG_CONVERTER_CANDIDATES"] = old_candidates
                else:
                    os.environ.pop("GEOFEM_DWG_CONVERTER_CANDIDATES", None)
                if old_auto is not None:
                    os.environ["GEOFEM_DWG_AUTODISCOVER"] = old_auto
                else:
                    os.environ.pop("GEOFEM_DWG_AUTODISCOVER", None)

    def test_dwg_file_without_converter_reports_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dwg = Path(tmp) / "model.dwg"
            dwg.write_bytes(b"fake-dwg")
            old_auto = os.environ.get("GEOFEM_DWG_AUTODISCOVER")
            old_converter = os.environ.pop("GEOFEM_DWG_CONVERTER", None)
            try:
                os.environ["GEOFEM_DWG_AUTODISCOVER"] = "0"
                with self.assertRaisesRegex(CadImportError, "GEOFEM_DWG_CONVERTER"):
                    parse_cad_file(dwg)
            finally:
                if old_auto is not None:
                    os.environ["GEOFEM_DWG_AUTODISCOVER"] = old_auto
                else:
                    os.environ.pop("GEOFEM_DWG_AUTODISCOVER", None)
                if old_converter is not None:
                    os.environ["GEOFEM_DWG_CONVERTER"] = old_converter


class QuadDominantMeshTests(unittest.TestCase):
    def test_rectangle_quad_dominant_mesh_keeps_all_quads_and_prunes_nodes(self) -> None:
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 1.0),
            target=1.0,
            regions=[],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(len(mesh["elements"]), 2)
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(len(mesh["nodes"]), 6)
        self.assertEqual(mesh["mesh_quality"]["quad_ratio"], 1.0)

    def test_quad_dominant_mesh_respects_region_centers(self) -> None:
        region = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 2.0),
            target=1.0,
            regions=[region],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertGreaterEqual(len(mesh["elements"]), 3)
        self.assertEqual(mesh["mode"], "rectilinear_polygon_paving")
        self.assertEqual(mesh["mesh_quality"]["rectilinear_paving_block_count"], 3)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [region], []))

    def test_mapped_paving_quad_block_follows_skew_quadrilateral_boundary(self) -> None:
        region = [(0.0, 0.0), (2.0, 0.0), (2.3, 1.0), (0.2, 1.0)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.3, 0.0, 1.0),
            target=0.5,
            regions=[region],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "mapped_quad_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["mapped_paving_quad_count"], len(mesh["elements"]))
        self.assertGreaterEqual(mesh["mesh_quality"]["min_quad_angle_deg"], 60.0)
        left_edge = (region[0], region[3])
        right_edge = (region[1], region[2])
        for nid in mesh["node_sets"]["left"]:
            self.assertLess(point_to_segment_distance(tuple(mesh["nodes"][nid]), *left_edge), 1.0e-10)
        for nid in mesh["node_sets"]["right"]:
            self.assertLess(point_to_segment_distance(tuple(mesh["nodes"][nid]), *right_edge), 1.0e-10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [region], []))

    def test_mapped_paving_adjacent_quadrilateral_regions_share_interface_nodes(self) -> None:
        left = [(0.0, 0.0), (1.2, 0.1), (1.0, 1.1), (0.0, 1.0)]
        right = [(1.2, 0.1), (2.3, 0.0), (2.1, 1.0), (1.0, 1.1)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.3, 0.0, 1.1),
            target=0.5,
            regions=[left, right],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "mapped_quad_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["mapped_paving_region_count"], 2)
        self.assertIn("region_1", mesh["element_sets"])
        self.assertIn("region_2", mesh["element_sets"])
        region_1 = set(mesh["element_sets"]["region_1"])
        region_2 = set(mesh["element_sets"]["region_2"])
        node_regions = {nid: set() for nid in mesh["nodes"]}
        for element in mesh["elements"]:
            region = "region_1" if element["id"] in region_1 else "region_2" if element["id"] in region_2 else ""
            for nid in element["nodes"]:
                node_regions[str(nid)].add(region)
        interface = [
            nid
            for nid, point in mesh["nodes"].items()
            if point_to_segment_distance(tuple(point), left[1], left[2]) <= 1.0e-9
        ]
        self.assertGreaterEqual(len(interface), 3)
        self.assertTrue(all({"region_1", "region_2"}.issubset(node_regions[nid]) for nid in interface))
        self.assertGreaterEqual(mesh["mesh_quality"]["mapped_paving_shared_node_count"], len(interface))

    def test_quad8_generation_shares_midside_nodes_between_adjacent_regions(self) -> None:
        left = [(0.0, 0.0), (1.2, 0.1), (1.0, 1.1), (0.0, 1.0)]
        right = [(1.2, 0.1), (2.3, 0.0), (2.1, 1.0), (1.0, 1.1)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.3, 0.0, 1.1),
            target=0.5,
            regions=[left, right],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD8",
        )
        self.assertEqual(mesh["mode"], "mapped_quad_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD8"})
        self.assertGreater(mesh["mesh_quality"]["promoted_midside_node_count"], 0)
        region_1 = set(mesh["element_sets"]["region_1"])
        region_2 = set(mesh["element_sets"]["region_2"])
        node_regions = {nid: set() for nid in mesh["nodes"]}
        for element in mesh["elements"]:
            region = "region_1" if element["id"] in region_1 else "region_2" if element["id"] in region_2 else ""
            for nid in element["nodes"]:
                node_regions[str(nid)].add(region)
        interface = [
            nid
            for nid, point in mesh["nodes"].items()
            if point_to_segment_distance(tuple(point), left[1], left[2]) <= 1.0e-9
        ]
        self.assertGreaterEqual(len(interface), 5)
        self.assertTrue(all({"region_1", "region_2"}.issubset(node_regions[nid]) for nid in interface))

    def test_rectilinear_polygon_paving_decomposes_l_shape_into_quad_blocks(self) -> None:
        region = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]
        self.assertIsNotNone(normalize_rectilinear_region(region))
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 3.0, 0.0, 3.0),
            target=0.5,
            regions=[region],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "rectilinear_polygon_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["rectilinear_paving_block_count"], 3)
        self.assertEqual(mesh["mesh_quality"]["rectilinear_paving_quad_count"], len(mesh["elements"]))
        self.assertGreater(mesh["mesh_quality"]["rectilinear_paving_shared_node_count"], 0)
        self.assertGreaterEqual(mesh["mesh_quality"]["min_quad_angle_deg"], 89.0)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [region], []))

    def test_rectilinear_polygon_paving_handles_multiple_outer_regions(self) -> None:
        left = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        right = [(2.0, 0.0), (4.0, 0.0), (4.0, 1.0), (2.0, 1.0)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 1.0),
            target=0.5,
            regions=[left, right],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "rectilinear_polygon_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["rectilinear_paving_region_count"], 2)
        self.assertEqual(mesh["mesh_quality"]["rectilinear_paving_quad_count"], len(mesh["elements"]))
        self.assertIn("region_1", mesh["element_sets"])
        self.assertIn("region_2", mesh["element_sets"])
        self.assertGreater(len(mesh["element_sets"]["region_2"]), len(mesh["element_sets"]["region_1"]))
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [left, right], []))
            self.assertFalse(1.0 < cx < 2.0)

    def test_rectilinear_paving_rejects_oblique_polygon(self) -> None:
        region = [(0.0, 0.0), (2.0, 0.0), (2.2, 1.0), (0.0, 1.0)]
        self.assertIsNone(normalize_rectilinear_region(region))
        self.assertIsNone(
            generate_rectilinear_polygon_paving_mesh(
                bbox=(0.0, 2.2, 0.0, 1.0),
                target=0.5,
                regions=[region],
                tunnels=[],
                material="soil",
                integration="FULL",
                requested_type="QUAD4",
            )
        )

    def test_simple_polygon_quad_paving_handles_oblique_concave_region(self) -> None:
        region = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.2), (1.8, 0.9), (3.0, 2.0), (0.0, 2.0), (0.8, 1.0)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 3.0, 0.0, 2.0),
            target=1.0,
            regions=[region],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "simple_polygon_quad_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_quad_paving_triangle_count"], len(region) - 2)
        self.assertEqual(mesh["mesh_quality"]["polygon_quad_paving_quad_count"], len(mesh["elements"]))
        self.assertGreater(mesh["mesh_quality"]["polygon_quad_paving_concave_vertex_count"], 0)
        self.assertGreaterEqual(mesh["mesh_quality"]["min_quad_angle_deg"], 10.0)
        segments = list(zip(region, region[1:] + region[:1]))
        for nid in mesh["node_sets"]["boundary"]:
            point = tuple(mesh["nodes"][nid])
            self.assertLess(min(point_to_segment_distance(point, a, b) for a, b in segments), 1.0e-10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [region], []))

    def test_simple_polygon_quad_paving_handles_multiple_oblique_regions(self) -> None:
        left = [(0.0, 0.0), (1.4, 0.1), (1.2, 1.0), (0.0, 0.9)]
        right = [(2.0, 0.0), (3.5, 0.2), (3.2, 1.1), (2.1, 0.9)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 3.5, 0.0, 1.1),
            target=0.5,
            regions=[left, right],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "simple_polygon_quad_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_quad_paving_region_count"], 2)
        self.assertEqual(mesh["mesh_quality"]["polygon_quad_paving_triangle_count"], 4)
        self.assertEqual(mesh["mesh_quality"]["polygon_quad_paving_quad_count"], len(mesh["elements"]))
        self.assertIn("region_1", mesh["element_sets"])
        self.assertIn("region_2", mesh["element_sets"])
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [left, right], []))
            self.assertFalse(1.4 < cx < 2.0)

    def test_rectilinear_polygon_paving_preserves_rectangular_hole(self) -> None:
        outer = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
        hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 3.0, 0.0, 3.0),
            target=1.0,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "rectilinear_polygon_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_count"], 1)
        self.assertEqual(mesh["mesh_quality"]["rectilinear_paving_block_count"], 8)
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertIn("hole_boundary", mesh["node_sets"])
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=[hole]))
            self.assertFalse(1.0 < cx < 2.0 and 1.0 < cy < 2.0)

    def test_quadrilateral_ring_paving_preserves_oblique_hole_boundary(self) -> None:
        outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        hole = [(1.0, 0.8), (2.8, 1.0), (2.5, 2.1), (0.9, 1.8)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.0),
            target=0.75,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "quadrilateral_ring_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_count"], 1)
        self.assertEqual(mesh["mesh_quality"]["quadrilateral_ring_paving_patch_count"], 4)
        self.assertEqual(mesh["mesh_quality"]["quadrilateral_ring_paving_quad_count"], len(mesh["elements"]))
        self.assertGreaterEqual(mesh["mesh_quality"]["min_quad_angle_deg"], 10.0)
        hole_segments = list(zip(hole, hole[1:] + hole[:1]))
        for nid in mesh["node_sets"]["hole_boundary"]:
            point = tuple(mesh["nodes"][nid])
            self.assertLess(min(point_to_segment_distance(point, a, b) for a, b in hole_segments), 1.0e-10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=[hole]))

    def test_polygon_hole_paving_preserves_oblique_pentagonal_hole_boundary(self) -> None:
        outer = [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]
        hole = [(2.0, 1.0), (3.2, 1.2), (3.4, 2.2), (2.5, 2.9), (1.6, 2.1)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 5.0, 0.0, 4.0),
            target=0.8,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "quadrilateral_polygon_hole_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_count"], 1)
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_paving_patch_count"], len(hole))
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_paving_quad_count"], len(mesh["elements"]))
        self.assertGreaterEqual(mesh["mesh_quality"]["min_quad_angle_deg"], 8.0)
        hole_segments = list(zip(hole, hole[1:] + hole[:1]))
        for nid in mesh["node_sets"]["hole_boundary"]:
            point = tuple(mesh["nodes"][nid])
            self.assertLess(min(point_to_segment_distance(point, a, b) for a, b in hole_segments), 1.0e-10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=[hole]))

    def test_polygon_hole_paving_handles_star_shaped_concave_hole(self) -> None:
        outer = [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)]
        hole = [(2.0, 1.0), (4.0, 1.0), (4.0, 3.0), (3.1, 2.2), (2.0, 3.2)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 6.0, 0.0, 5.0),
            target=0.8,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "quadrilateral_polygon_hole_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_paving_concave_vertex_count"], 1)
        self.assertTrue(mesh["mesh_quality"]["polygon_hole_paving_star_shaped"])
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_paving_quad_count"], len(mesh["elements"]))
        hole_segments = list(zip(hole, hole[1:] + hole[:1]))
        for nid in mesh["node_sets"]["hole_boundary"]:
            point = tuple(mesh["nodes"][nid])
            self.assertLess(min(point_to_segment_distance(point, a, b) for a, b in hole_segments), 1.0e-10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=[hole]))

    def test_non_star_concave_polygon_hole_boundary_grid_stays_all_quad(self) -> None:
        outer = [(0.0, 0.0), (7.0, 0.0), (7.0, 5.0), (0.0, 5.0)]
        hole = [
            (2.0, 1.0),
            (4.2, 1.1),
            (4.1, 1.55),
            (2.75, 1.45),
            (2.65, 2.45),
            (4.0, 2.55),
            (4.1, 3.0),
            (2.0, 3.1),
        ]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 7.0, 0.0, 5.0),
            target=0.5,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "non_star_polygon_hole_boundary_grid")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["remaining_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_non_star_count"], 1)
        self.assertGreater(mesh["mesh_quality"]["polygon_hole_total_concave_vertex_count"], 0)
        self.assertIn("hole_boundary", mesh["node_sets"])
        hole_segments = list(zip(hole, hole[1:] + hole[:1]))
        for nid in mesh["node_sets"]["hole_boundary"]:
            point = tuple(mesh["nodes"][nid])
            self.assertLess(min(point_to_segment_distance(point, a, b) for a, b in hole_segments), 1.0e-10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=[hole]))

    def test_gf1_sampled_curve_region_and_hole_mesh_as_all_quads(self) -> None:
        text = (
            '{"geometry":{"regions":[{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,3]},'
            '{"type":"bezier","points":[[4,3],[3,3.5],[1,3.5],[0,3]],"segments":6},'
            '{"type":"line","start":[0,3],"end":[0,0]}'
            '],"holes":[{"type":"circle","center":[2,1.5],"radius":0.35,"segments":24}]}]}}'
        )
        doc = parse_gf1_document(text)
        outer = [tuple(point) for point in doc["regions"][0]["points"]]
        hole = [tuple(point) for point in doc["regions"][0]["holes"][0]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.4),
            target=0.45,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["remaining_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_boundary_segment_count"], 24)
        self.assertGreater(mesh["mesh_quality"]["oblique_boundary_layer_added_line_count"], 0)
        self.assertIn("hole_boundary", mesh["node_sets"])
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=[hole]))

    def test_parametric_curved_quad_paving_keeps_bezier_boundary(self) -> None:
        text = (
            '{"geometry":{"regions":[{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,3]},'
            '{"type":"bezier","points":[[4,3],[3,3.7],[1,3.7],[0,3]]},'
            '{"type":"line","start":[0,3],"end":[0,0]}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        region = doc["regions"][0]
        outer = [tuple(point) for point in region["points"]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.8),
            target=0.5,
            regions=[outer],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=[region],
        )
        self.assertEqual(mesh["mode"], "parametric_curved_quad_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["curve_parameter_retained"])
        self.assertEqual(mesh["mesh_quality"]["parametric_curved_quad_paving_curve_count"], 4)
        top_y = [mesh["nodes"][nid][1] for nid in mesh["node_sets"]["top"]]
        self.assertGreater(max(top_y), 3.2)
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)

    def test_multi_parametric_curved_outer_boundaries_split_into_blocks(self) -> None:
        text = (
            '{"geometry":{"regions":['
            '{"boundary":['
            '{"type":"line","start":[0,0],"end":[3,0]},'
            '{"type":"line","start":[3,0],"end":[3,2.5]},'
            '{"type":"bezier","points":[[3,2.5],[2.4,3.2],[0.6,3.2],[0,2.5]]},'
            '{"type":"line","start":[0,2.5],"end":[0,0]}'
            ']},'
            '{"boundary":['
            '{"type":"line","start":[4,0],"end":[7,0]},'
            '{"type":"line","start":[7,0],"end":[7,2.5]},'
            '{"type":"bezier","points":[[7,2.5],[6.4,3.1],[4.6,3.1],[4,2.5]]},'
            '{"type":"line","start":[4,2.5],"end":[4,0]}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        regions = [[tuple(point) for point in region["points"]] for region in doc["regions"]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 7.0, 0.0, 3.3),
            target=0.5,
            regions=regions,
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=doc["regions"],
        )
        self.assertEqual(mesh["mode"], "multi_parametric_curved_outer_block_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["curve_parameter_retained"])
        self.assertEqual(mesh["mesh_quality"]["parametric_curved_outer_block_count"], 2)
        self.assertEqual(len(mesh["mesh_quality"]["parametric_curved_outer_boundary_node_counts"]), 2)
        self.assertGreaterEqual(min(mesh["mesh_quality"]["parametric_curved_outer_boundary_node_counts"]), 20)
        self.assertIn("curved_outer_boundary_1", mesh["node_sets"])
        self.assertIn("curved_outer_boundary_2", mesh["node_sets"])
        self.assertIn("region_1", mesh["element_sets"])
        self.assertIn("region_2", mesh["element_sets"])
        self.assertGreater(max(mesh["nodes"][nid][1] for nid in mesh["node_sets"]["curved_outer_boundary_1"]), 2.8)
        self.assertGreater(max(mesh["nodes"][nid][1] for nid in mesh["node_sets"]["curved_outer_boundary_2"]), 2.7)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, regions, []))
            self.assertFalse(3.05 < cx < 3.95)

    def test_touching_parametric_curved_outer_boundaries_share_nodes(self) -> None:
        text = (
            '{"geometry":{"regions":['
            '{"boundary":['
            '{"type":"line","start":[0,0],"end":[3,0]},'
            '{"type":"line","start":[3,0],"end":[3,2.5]},'
            '{"type":"bezier","points":[[3,2.5],[2.4,3.2],[0.6,3.2],[0,2.5]]},'
            '{"type":"line","start":[0,2.5],"end":[0,0]}'
            ']},'
            '{"boundary":['
            '{"type":"line","start":[3,0],"end":[6,0]},'
            '{"type":"line","start":[6,0],"end":[6,2.5]},'
            '{"type":"bezier","points":[[6,2.5],[5.4,3.1],[3.6,3.1],[3,2.5]]},'
            '{"type":"line","start":[3,2.5],"end":[3,0]}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        regions = [[tuple(point) for point in region["points"]] for region in doc["regions"]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 6.0, 0.0, 3.3),
            target=0.5,
            regions=regions,
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=doc["regions"],
        )
        self.assertEqual(mesh["mode"], "multi_parametric_curved_outer_block_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["parametric_curved_outer_boolean_repaired"])
        self.assertGreater(mesh["mesh_quality"]["parametric_curved_outer_shared_boundary_node_count"], 0)
        self.assertGreater(mesh["mesh_quality"]["parametric_curved_outer_merged_node_count"], 0)
        shared = set(mesh["node_sets"]["shared_curved_outer_boundary"])
        self.assertTrue(shared)
        self.assertTrue(shared.issubset(set(mesh["node_sets"]["curved_outer_boundary_1"])))
        self.assertTrue(shared.issubset(set(mesh["node_sets"]["curved_outer_boundary_2"])))
        for nid in shared:
            x, _y = mesh["nodes"][nid]
            self.assertAlmostEqual(x, 3.0, places=8)
        self.assertEqual(len(mesh["nodes"]), len({tuple(point) for point in mesh["nodes"].values()}))
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, regions, []))

    def test_overlapping_parametric_curved_outer_boundaries_boolean_union(self) -> None:
        text = (
            '{"geometry":{"regions":['
            '{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,2.5]},'
            '{"type":"bezier","points":[[4,2.5],[3.2,3.2],[0.8,3.2],[0,2.5]]},'
            '{"type":"line","start":[0,2.5],"end":[0,0]}'
            ']},'
            '{"boundary":['
            '{"type":"line","start":[3,0],"end":[7,0]},'
            '{"type":"line","start":[7,0],"end":[7,2.5]},'
            '{"type":"bezier","points":[[7,2.5],[6.2,3.1],[3.8,3.1],[3,2.5]]},'
            '{"type":"line","start":[3,2.5],"end":[3,0]}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        regions = [[tuple(point) for point in region["points"]] for region in doc["regions"]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 7.0, 0.0, 3.3),
            target=0.5,
            regions=regions,
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=doc["regions"],
        )
        self.assertEqual(mesh["mode"], "boolean_repaired_curved_outer_union_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["parametric_curved_outer_boolean_repaired"])
        self.assertEqual(mesh["mesh_quality"]["boolean_engine"], "analytic_curve_graph")
        self.assertEqual(mesh["mesh_quality"]["boolean_mesh_boundary_source"], "analytic_curve_graph_loops")
        self.assertEqual(mesh["mesh_quality"]["boolean_element_set_classifier"], "analytic_curve_graph_winding")
        self.assertEqual(mesh["mesh_quality"]["boolean_overlap_area_source"], "analytic_intersection_loop_area")
        self.assertGreater(mesh["mesh_quality"]["boolean_overlap_area"], 0.0)
        self.assertGreater(mesh["mesh_quality"]["boolean_overlap_element_count"], 0)
        self.assertIn("boolean_overlap", mesh["element_sets"])
        self.assertIn("region_1", mesh["element_sets"])
        self.assertIn("region_2", mesh["element_sets"])
        self.assertIn("boolean_region_1_difference", mesh["element_sets"])
        self.assertIn("boolean_region_2_difference", mesh["element_sets"])
        self.assertIn("boolean_boundary", mesh["node_sets"])
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, regions, []))

    def test_boolean_expression_mesh_uses_analytic_graph_loop_domain(self) -> None:
        region_1 = {
            "points": [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
            "boundary": [
                {"type": "line", "start": [0.0, 0.0], "end": [3.0, 0.0]},
                {"type": "line", "start": [3.0, 0.0], "end": [3.0, 2.0]},
                {"type": "line", "start": [3.0, 2.0], "end": [0.0, 2.0]},
                {"type": "line", "start": [0.0, 2.0], "end": [0.0, 0.0]},
            ],
        }
        region_2 = {
            "points": [[2.0, 0.5], [5.0, 0.5], [5.0, 2.5], [2.0, 2.5]],
            "boundary": [
                {"type": "line", "start": [2.0, 0.5], "end": [5.0, 0.5]},
                {"type": "line", "start": [5.0, 0.5], "end": [5.0, 2.5]},
                {"type": "line", "start": [5.0, 2.5], "end": [2.0, 2.5]},
                {"type": "line", "start": [2.0, 2.5], "end": [2.0, 0.5]},
            ],
        }
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 5.0, 0.0, 2.5),
            target=0.25,
            regions=[[tuple(point) for point in region_1["points"]], [tuple(point) for point in region_2["points"]]],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=[region_1, region_2],
            boolean_expression="A-B",
        )
        self.assertEqual(mesh["mode"], "boolean_repaired_curved_outer_expression_paving")
        self.assertEqual(mesh["cad_boolean"]["selected_operation"], "expression")
        self.assertEqual(mesh["cad_boolean"]["boolean_expression"], "A-B")
        self.assertEqual(mesh["mesh_quality"]["boolean_mesh_boundary_source"], "analytic_curve_graph_loops")
        nodes = mesh["nodes"]
        region_1_points = [tuple(point) for point in region_1["points"]]
        region_2_points = [tuple(point) for point in region_2["points"]]
        for element in mesh["elements"]:
            pts = [nodes[nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [region_1_points], []))
            self.assertFalse(inside_domain(cx, cy, [region_2_points], []))

    def test_self_intersecting_parametric_curved_outer_boundary_is_repaired(self) -> None:
        region = {
            "boundary": [
                {"type": "line", "start": [0.0, 0.0], "end": [4.0, 3.0]},
                {"type": "line", "start": [4.0, 3.0], "end": [0.0, 3.0]},
                {"type": "line", "start": [0.0, 3.0], "end": [4.0, 0.0]},
                {"type": "line", "start": [4.0, 0.0], "end": [0.0, 0.0]},
            ],
        }
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.0),
            target=0.5,
            regions=[],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=[region],
        )
        self.assertEqual(mesh["mode"], "boolean_repaired_curved_outer_union_paving")
        self.assertTrue(mesh["mesh_quality"]["parametric_curved_outer_boolean_repaired"])
        self.assertEqual(mesh["mesh_quality"]["boolean_self_intersection_repaired_count"], 1)
        self.assertEqual(mesh["mesh_quality"]["boolean_self_intersection_repair"], "analytic_curve_graph_loops")
        self.assertEqual(mesh["mesh_quality"]["boolean_mesh_boundary_source"], "analytic_curve_graph_loops")
        self.assertGreaterEqual(mesh["mesh_quality"]["boolean_union_region_count"], 1)
        self.assertGreater(len(mesh["elements"]), 0)
        repaired_parts = [
            [(0.0, 0.0), (2.0, 1.5), (4.0, 0.0)],
            [(4.0, 3.0), (2.0, 1.5), (0.0, 3.0)],
        ]
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, repaired_parts, []))

    def test_boolean_curved_outer_preserves_analytic_curve_trims(self) -> None:
        text = (
            '{"geometry":{"regions":['
            '{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,2.5]},'
            '{"type":"bezier","control_points":[[4,2.5],[3.2,3.25],[0.8,3.25],[0,2.5]]},'
            '{"type":"arc","center":[0,1.25],"radius":1.25,"start_angle":90,"end_angle":270}'
            ']},'
            '{"boundary":['
            '{"type":"line","start":[3,0],"end":[7,0]},'
            '{"type":"line","start":[7,0],"end":[7,2.5]},'
            '{"type":"nurbs","control_points":[[7,2.5],[6,3.1],[4,3.1],[3,2.5]],"weights":[1,0.9,0.9,1],"knots":[0,0,0,0,1,1,1,1],"degree":3},'
            '{"type":"line","start":[3,2.5],"end":[3,0]}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        regions = [[tuple(point) for point in region["points"]] for region in doc["regions"]]
        mesh = generate_quad_dominant_mesh(
            bbox=(-1.3, 7.0, 0.0, 3.4),
            target=0.45,
            regions=regions,
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=doc["regions"],
        )
        self.assertEqual(mesh["mode"], "boolean_repaired_curved_outer_union_paving")
        self.assertEqual(mesh["cad_boolean"]["engine"], "analytic_curve_graph_winding_containment")
        self.assertEqual(mesh["cad_boolean"]["area_selection_engine"], "analytic_winding_containment")
        analytic_segments = mesh["cad_boolean"]["analytic_boundary_segments"]
        analytic_graph = mesh["cad_boolean"]["analytic_curve_graph"]
        retained_types = {segment["type"] for segment in analytic_segments}
        self.assertTrue({"arc", "bezier", "nurbs"}.issubset(retained_types))
        self.assertEqual(analytic_graph["selection_engine"], "analytic_winding_containment")
        self.assertGreater(analytic_graph["intersection_count"], 0)
        self.assertGreater(analytic_graph["split_edge_count"], analytic_graph["curve_count"])
        self.assertGreater(len(analytic_graph["union_loops"]), 0)
        self.assertIn("intersection", analytic_graph["boolean_operations"])
        self.assertTrue(any(edge.get("graph_union_boundary") for edge in analytic_graph["edges"]))
        self.assertGreater(mesh["mesh_quality"]["boolean_analytic_curve_retained_count"], 0)
        self.assertGreater(mesh["mesh_quality"]["boolean_analytic_trimmed_curve_count"], 0)
        self.assertGreater(mesh["mesh_quality"]["boolean_analytic_intersection_count"], 0)
        nurbs_segments = [segment for segment in analytic_segments if segment["type"] == "nurbs"]
        self.assertTrue(nurbs_segments)
        self.assertEqual(nurbs_segments[0]["source"]["degree"], 3)
        self.assertEqual(nurbs_segments[0]["source"]["weights"], [1.0, 0.9, 0.9, 1.0])

    def test_parametric_curve_hole_paving_keeps_circle_nodes_on_radius(self) -> None:
        text = (
            '{"geometry":{"regions":[{"points":[[0,0],[4,0],[4,3],[0,3]],'
            '"holes":[{"type":"circle","center":[2,1.5],"radius":0.4}]}]}}'
        )
        doc = parse_gf1_document(text)
        region = doc["regions"][0]
        outer = [tuple(point) for point in region["points"]]
        hole = [tuple(point) for point in region["holes"][0]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.0),
            target=0.45,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=[region],
        )
        self.assertEqual(mesh["mode"], "parametric_curve_hole_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["curve_parameter_retained"])
        self.assertEqual(mesh["mesh_quality"]["parametric_curve_hole_paving_curve_count"], 1)
        self.assertIn("hole_boundary", mesh["node_sets"])
        for nid in mesh["node_sets"]["hole_boundary"]:
            x, y = mesh["nodes"][nid]
            self.assertAlmostEqual(math.hypot(x - 2.0, y - 1.5), 0.4, places=8)
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)

    def test_parametric_curve_hole_paving_accepts_curved_outer_boundary(self) -> None:
        text = (
            '{"geometry":{"regions":[{"boundary":['
            '{"type":"line","start":[0,0],"end":[4,0]},'
            '{"type":"line","start":[4,0],"end":[4,3]},'
            '{"type":"bezier","points":[[4,3],[3,3.5],[1,3.5],[0,3]]},'
            '{"type":"line","start":[0,3],"end":[0,0]}'
            '],"holes":[{"type":"circle","center":[2,1.5],"radius":0.35}]}]}}'
        )
        doc = parse_gf1_document(text)
        region = doc["regions"][0]
        outer = [tuple(point) for point in region["points"]]
        hole = [tuple(point) for point in region["holes"][0]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.5),
            target=0.45,
            regions=[outer],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=[region],
        )
        self.assertEqual(mesh["mode"], "parametric_curve_hole_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["curve_parameter_retained"])
        self.assertGreaterEqual(mesh["mesh_quality"]["parametric_curve_hole_paving_patch_count"], 8)
        self.assertGreater(max(mesh["nodes"][nid][1] for nid in mesh["node_sets"]["outer_boundary"]), 3.2)

    def test_multi_parametric_curve_holes_keep_each_circle_boundary(self) -> None:
        text = (
            '{"geometry":{"regions":[{"points":[[0,0],[6,0],[6,3],[0,3]],'
            '"holes":['
            '{"type":"circle","center":[2,1.5],"radius":0.35},'
            '{"type":"circle","center":[4,1.5],"radius":0.35}'
            ']}]}}'
        )
        doc = parse_gf1_document(text)
        region = doc["regions"][0]
        outer = [tuple(point) for point in region["points"]]
        holes = [[tuple(point) for point in hole] for hole in region["holes"]]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 6.0, 0.0, 3.0),
            target=0.4,
            regions=[outer],
            tunnels=[],
            polygon_holes=holes,
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            curved_regions=[region],
        )
        self.assertEqual(mesh["mode"], "multi_parametric_curve_hole_boundary_grid")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertTrue(mesh["mesh_quality"]["curve_parameter_retained"])
        self.assertEqual(mesh["mesh_quality"]["parametric_curve_hole_count"], 2)
        self.assertEqual(len(mesh["mesh_quality"]["parametric_curve_hole_boundary_node_counts"]), 2)
        for index, (cx, cy, radius) in enumerate([(2.0, 1.5, 0.35), (4.0, 1.5, 0.35)], start=1):
            key = f"parametric_hole_boundary_{index}"
            self.assertIn(key, mesh["node_sets"])
            self.assertGreater(len(mesh["node_sets"][key]), 0)
            for nid in mesh["node_sets"][key]:
                x, y = mesh["nodes"][nid]
                self.assertAlmostEqual(math.hypot(x - cx, y - cy), radius, places=8)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [outer], [], polygon_holes=holes))

    def test_quad_grid_excludes_polygon_hole_when_no_outer_region_is_given(self) -> None:
        hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 3.0, 0.0, 3.0),
            target=1.0,
            regions=[],
            tunnels=[],
            polygon_holes=[hole],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mesh_quality"]["polygon_hole_count"], 1)
        self.assertEqual(mesh["mesh_quality"]["quad_count"], len(mesh["elements"]))
        self.assertGreaterEqual(mesh["mesh_quality"]["quad_count"], 8)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertFalse(1.0 < cx < 2.0 and 1.0 < cy < 2.0)

    def test_triangle_recombination_promotes_adjacent_tris_to_quality_quad(self) -> None:
        nodes = {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]}
        elements = [
            {"id": "1", "type": "TRI3", "nodes": ["1", "2", "3"], "material": "soil", "integration": "FULL"},
            {"id": "2", "type": "TRI3", "nodes": ["1", "3", "4"], "material": "soil", "integration": "FULL"},
        ]
        recombined, info = recombine_triangles_to_quads(nodes, elements)
        self.assertEqual(info["recombined_quad_count"], 1)
        self.assertEqual(info["remaining_tri_count"], 0)
        self.assertEqual(len(recombined), 1)
        self.assertEqual(recombined[0]["type"], "QUAD4")
        self.assertEqual(set(recombined[0]["nodes"]), {"1", "2", "3", "4"})

    def test_quad_dominant_mesh_reports_recombination_quality_fields(self) -> None:
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 1.0),
            target=1.0,
            regions=[],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        quality = mesh["mesh_quality"]
        self.assertIn("recombined_quad_count", quality)
        self.assertIn("min_quad_angle_deg", quality)
        self.assertIn("max_quad_aspect_ratio", quality)
        self.assertGreaterEqual(quality["min_quad_angle_deg"], 90.0)

    def test_boundary_projection_snaps_near_region_nodes_and_reports_quality(self) -> None:
        region = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.92), (0.0, 0.92)]
        nodes = {"1": [0.0, 1.0], "2": [1.0, 1.0], "3": [2.0, 1.0], "4": [1.0, 0.5]}
        projected, info = project_nodes_to_boundaries(nodes, (0.0, 2.0, 0.0, 1.0), [region], [], 0.5)
        y_values = {round(point[1], 6) for point in projected.values()}
        self.assertIn(0.92, y_values)
        self.assertEqual(info["projected_node_count"], 3)
        self.assertAlmostEqual(info["snap_tolerance"], 0.125)

    def test_boundary_layer_adds_near_region_lines_to_quad_mesh(self) -> None:
        region = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.92), (0.0, 0.92)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 1.0),
            target=0.5,
            regions=[region],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        y_values = {round(point[1], 6) for point in mesh["nodes"].values()}
        self.assertIn(0.92, y_values)
        self.assertGreater(mesh["mesh_quality"]["boundary_layer_added_y_count"], 0)
        self.assertLessEqual(mesh["mesh_quality"]["max_quad_aspect_ratio"], 4.0)

    def test_quad_mesh_honors_local_refinement_lines(self) -> None:
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 2.0),
            target=1.0,
            regions=[],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            refinements=[{"center": [1.0, 1.0], "radius": 0.5, "factor": 2.0}],
        )
        self.assertGreater(len(mesh["elements"]), 4)
        self.assertEqual(mesh["mesh_quality"]["local_refinement_added_line_count"], 4)
        self.assertEqual(mesh["mesh_quality"]["x_line_count"], 5)
        self.assertEqual(mesh["mesh_quality"]["y_line_count"], 5)

    def test_quad_mesh_honors_split_line_constraints(self) -> None:
        split = {"type": "split_line", "start": [1.0, 0.0], "end": [1.0, 2.0], "target_size": 0.5}
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 2.0),
            target=1.0,
            regions=[],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
            refinements=[split],
        )
        self.assertEqual(normalize_mesh_split_lines([split]), [((1.0, 0.0), (1.0, 2.0), 0.5)])
        self.assertEqual(mesh["mesh_quality"]["split_line_constraint_count"], 1)
        self.assertGreaterEqual(mesh["mesh_quality"]["split_line_added_line_count"], 1)
        x_values = {round(point[0], 6) for point in mesh["nodes"].values()}
        self.assertIn(1.0, x_values)

    def test_mesh_quality_improvement_compares_smoothing_and_remesh(self) -> None:
        nodes = {
            "1": [0.0, 0.0],
            "2": [1.0, 0.0],
            "3": [2.0, 0.0],
            "4": [0.0, 1.0],
            "5": [1.7, 0.2],
            "6": [2.0, 1.0],
            "7": [0.0, 2.0],
            "8": [1.0, 2.0],
            "9": [2.0, 2.0],
        }
        elements = [
            {"id": "1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"]},
            {"id": "2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"]},
            {"id": "3", "type": "QUAD4", "nodes": ["4", "5", "8", "7"]},
            {"id": "4", "type": "QUAD4", "nodes": ["5", "6", "9", "8"]},
        ]
        smooth_nodes, smooth_elements, smooth_report = improve_mesh_quality(nodes, elements, method="laplace", min_angle_deg=30.0, max_aspect_ratio=5.0, max_skew=0.8, iterations=4)
        self.assertLess(smooth_report["after_violation_count"], smooth_report["before_violation_count"])
        self.assertGreater(smooth_report["changed_nodes"], 0)
        self.assertEqual(len(smooth_elements), len(elements))
        remesh_nodes, remesh_elements, remesh_report = improve_mesh_quality(nodes, elements, method="local_remesh", min_angle_deg=30.0, max_aspect_ratio=5.0, max_skew=0.8, iterations=1)
        self.assertGreater(len(remesh_nodes), len(nodes))
        self.assertGreater(len(remesh_elements), len(elements))
        self.assertGreater(remesh_report["changed_elements"], 0)

    def test_quadrilateral_circular_tunnel_paving_keeps_tunnel_nodes_on_arc(self) -> None:
        outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        tunnel = (2.0, 1.5, 0.45)
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 4.0, 0.0, 3.0),
            target=0.5,
            regions=[outer],
            tunnels=[tunnel],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "quadrilateral_circular_tunnel_paving")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["circular_tunnel_paving_patch_count"], 4)
        self.assertEqual(mesh["mesh_quality"]["circular_tunnel_paving_quad_count"], len(mesh["elements"]))
        self.assertGreaterEqual(mesh["mesh_quality"]["curved_boundary_segment_count"], curve_segment_count(tunnel[2], 0.5))
        self.assertIn("tunnel_boundary", mesh["node_sets"])
        cx, cy, radius = tunnel
        for nid in mesh["node_sets"]["tunnel_boundary"]:
            x, y = mesh["nodes"][nid]
            self.assertAlmostEqual(math.hypot(x - cx, y - cy), radius, places=10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            ex = sum(point[0] for point in pts) / len(pts)
            ey = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(ex, ey, [outer], [tunnel]))

    def test_multiple_circular_tunnels_quadify_and_tag_each_arc(self) -> None:
        outer = [(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (0.0, 3.0)]
        tunnels = [(2.0, 1.5, 0.35), (4.0, 1.5, 0.35)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 6.0, 0.0, 3.0),
            target=0.5,
            regions=[outer],
            tunnels=tunnels,
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertEqual(mesh["mode"], "multi_circular_tunnel_boundary_grid")
        self.assertEqual({element["type"] for element in mesh["elements"]}, {"QUAD4"})
        self.assertEqual(mesh["mesh_quality"]["fallback_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["remaining_tri_count"], 0)
        self.assertEqual(mesh["mesh_quality"]["circular_tunnel_count"], 2)
        self.assertEqual(len(mesh["mesh_quality"]["circular_tunnel_boundary_node_counts"]), 2)
        self.assertGreater(mesh["mesh_quality"]["subdivided_triangle_count"], 0)
        self.assertIn("tunnel_boundary", mesh["node_sets"])
        for index, (cx, cy, radius) in enumerate(tunnels, start=1):
            node_key = f"tunnel_boundary_{index}"
            self.assertIn(node_key, mesh["node_sets"])
            self.assertGreater(len(mesh["node_sets"][node_key]), 0)
            for nid in mesh["node_sets"][node_key]:
                x, y = mesh["nodes"][nid]
                self.assertAlmostEqual(math.hypot(x - cx, y - cy), radius, places=10)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            ex = sum(point[0] for point in pts) / len(pts)
            ey = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(ex, ey, [outer], tunnels))

    def test_oblique_boundary_layer_adds_lines_and_preserves_element_centers(self) -> None:
        region = [(0.0, 0.0), (2.0, 0.0), (2.0, 0.82), (0.0, 0.92)]
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 1.0),
            target=0.5,
            regions=[region],
            tunnels=[],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        self.assertGreater(mesh["mesh_quality"]["oblique_boundary_layer_added_line_count"], 0)
        self.assertLessEqual(mesh["mesh_quality"]["max_quad_aspect_ratio"], 10.0)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [region], []))

    def test_tunnel_curve_refinement_reports_quality_segments(self) -> None:
        mesh = generate_quad_dominant_mesh(
            bbox=(0.0, 2.0, 0.0, 2.0),
            target=0.5,
            regions=[],
            tunnels=[(1.0, 1.0, 0.35)],
            material="soil",
            integration="FULL",
            requested_type="QUAD4",
        )
        quality = mesh["mesh_quality"]
        self.assertEqual(quality["curved_boundary_segment_count"], curve_segment_count(0.35, 0.5))
        self.assertGreaterEqual(quality["curved_boundary_segment_count"], 24)
        self.assertGreater(quality["curved_boundary_layer_added_line_count"], 0)
        for element in mesh["elements"]:
            pts = [mesh["nodes"][nid] for nid in element["nodes"]]
            cx = sum(point[0] for point in pts) / len(pts)
            cy = sum(point[1] for point in pts) / len(pts)
            self.assertTrue(inside_domain(cx, cy, [], [(1.0, 1.0, 0.35)]))


if __name__ == "__main__":
    unittest.main()
