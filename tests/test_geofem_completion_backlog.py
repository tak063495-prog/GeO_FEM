from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from geofem_app.audit_trail import build_project_audit_trail
from geofem_app.api_contracts import api_contract_catalog, validate_api_contract, write_api_contract_docs
from geofem_app.cli import run_commercial_quality, run_customization, run_encoding_audit, run_sample_projects, run_solve, run_upgrade_check, run_workspace_dashboard
from geofem_app.commercial_quality import run_commercial_quality_check, run_output_reliability_gate
from geofem_app.customization import (
    ORGANIZATION_PROFILE_SCHEMA,
    apply_organization_profile,
    default_organization_profile,
    project_template_catalog,
    validate_organization_profile,
    write_customization_artifacts,
)
from geofem_app.encoding_policy import audit_text_encoding, configure_utf8_console, write_encoding_audit
from geofem_app.failure_diagnostics import classify_failure
from geofem_app.failure_recovery import build_failure_recovery_plan, write_failure_recovery_plan
from geofem_app.fem2d_constraints import constraint_helper_contract
from geofem_app.fem2d_dynamic import dynamic_helper_contract
from geofem_app.fem2d_element_fast_paths import element_fast_path_contract
from geofem_app.fem2d_element_advanced_elastic_post import advanced_elastic_post_contract
from geofem_app.fem2d_element_advanced_strength_kernels import advanced_strength_element_kernel_contract
from geofem_app.fem2d_element_elastic_kernels import elastic_element_kernel_contract
from geofem_app.fem2d_element_elastic_post import element_elastic_post_contract
from geofem_app.fem2d_element_interpolation import element_interpolation_contract
from geofem_app.fem2d_element_j2dp_kernels import j2dp_element_kernel_contract
from geofem_app.fem2d_element_mohr_coulomb_kernels import mohr_coulomb_element_kernel_contract
from geofem_app.fem2d_element_numba_primitives import element_numba_primitives_contract
from geofem_app.fem2d_element_post_processing import element_post_processing_contract
from geofem_app.fem2d_element_result_rows import element_result_row_contract
from geofem_app.fem2d_element_state_output import element_state_output_contract
from geofem_app.fem2d_element_tension_cutoff_kernels import tension_cutoff_element_kernel_contract
from geofem_app.fem2d_hydro import hydro_helper_contract
from geofem_app.fem2d_hydro_iteration import hydro_iteration_contract
from geofem_app.fem2d_mpc import mpc_solver_contract
from geofem_app.fem2d_nonlinear_assembly import nonlinear_assembly_contract
from geofem_app.fem2d_pressure import pressure_assembly_contract
from geofem_app.fem2d_result_annotations import result_annotation_contract
from geofem_app.fem2d_solver_progress import solver_progress_contract
from geofem_app.fem2d_structural_assembly import structural_assembly_contract
from geofem_app.geofeas_seepage import import_external_seepage_results as import_seepage_direct
from geofem_app.geofeas_verification import import_external_seepage_results as import_seepage_facade
from geofem_app.gui.analysis_panel import analysis_panel_contract
from geofem_app.gui.boundary_load_controller import boundary_load_controller_contract
from geofem_app.gui.command_catalog import command_hierarchy, gui_command_catalog, validate_command_catalog, write_command_catalog
from geofem_app.gui.domain_panels import domain_panel_contract
from geofem_app.gui.geometry_controller import geometry_controller_contract
from geofem_app.gui.i18n import available_gui_message_keys, gui_message, gui_message_catalog, validate_gui_i18n_catalog, write_gui_i18n_catalog
from geofem_app.gui.material_controller import material_controller_contract
from geofem_app.gui.mesh_auto_controller import mesh_auto_controller_contract
from geofem_app.gui.mesh_controller import mesh_controller_contract
from geofem_app.gui.model_check_panel import model_check_panel_contract, summarize_model_check_issues
from geofem_app.gui.post_report_controller import _performance_result_summary
from geofem_app.gui.post_drawing_helpers import post_drawing_helper_contract
from geofem_app.gui.post_report_controller import post_report_controller_contract
from geofem_app.gui.result_table_routes import known_result_table_kinds, result_table_path
from geofem_app.gui.selection_controller import selection_controller_contract
from geofem_app.gui.stage_controller import stage_controller_contract
from geofem_app.gui.surface_texts import (
    gui_surface_message,
    surface_text_catalog,
    translate_surface_text,
    validate_surface_text_catalog,
    write_surface_text_catalog,
)
from geofem_app.gui.workflow_guidance import build_workflow_guidance, workflow_steps, write_workflow_guidance
from geofem_app.gui.workflow_panel import workflow_panel_action_groups, workflow_panel_layout_contract
from geofem_app.gui.workspace_contract import (
    menu_bar_contract,
    validate_workspace_contract,
    workspace_responsibility_contract,
    workspace_view_contract,
    write_workspace_contract,
)
from geofem_app.gui.yaml_panels import yaml_panel_contract
from geofem_app.html_report_utils import format_value, html_escape, kv_table, rel_link, table
from geofem_app.input_assistance import build_input_assistance_summary, input_assistance_template_catalog, write_input_assistance_artifacts
from geofem_app.input_diagnostics import diagnose_input_config
from geofem_app.large_model_operations import (
    build_large_model_operation_profile,
    query_elements_by_bbox,
    query_nodes_by_bbox,
    write_large_model_operation_artifacts,
)
from geofem_app.maintainability_audit import audit_maintainability, write_maintainability_audit
from geofem_app.material_models import material_form_schema, material_model_catalog, write_material_reports
from geofem_app.messages import available_message_keys, message
from geofem_app.mesh_quality import evaluate_mesh_quality, write_mesh_quality_report
from geofem_app.output_comparison import compare_result_cases
from geofem_app.performance_kpis import KPI_AREAS, build_result_performance_kpi_matrix
from geofem_app.performance_monitor import benchmark_case_performance, build_performance_summary, compare_performance_cases, write_performance_summary
from geofem_app.pdf_writer import write_text_pdf
from geofem_app.project import new_default_project, save_project
from geofem_app.samples import plane_strain_patch_sample, plane_strain_quad4_sample
from geofem_app.reliability_summary import build_reliability_summary
from geofem_app.sample_projects import build_sample_project_config, sample_project_catalog, write_sample_project_suite
from geofem_app.standard_benchmarks import run_standard_benchmark_suite
from geofem_app.startup_check import run_startup_check
from geofem_app.startup_support import SupportPackageOptions, write_startup_support_artifacts
from geofem_app.fem2d import mesh_from_config
from geofem_app.fem2d_solver import solve_plane_strain_config
from geofem_app.update_compatibility import build_update_compatibility_report, write_update_compatibility_artifacts
from geofem_app.version_info import build_version_info, write_version_info_artifacts
from geofem_app.workspace_management import WORKSPACE_ARCHIVE_SCHEMA, WORKSPACE_DASHBOARD_SCHEMA, build_workspace_dashboard, write_workspace_dashboard


class GeoFEMCompletionBacklogTests(unittest.TestCase):
    def test_analysis_panel_contract_is_split_from_main_window(self) -> None:
        contract = analysis_panel_contract()
        self.assertEqual(contract["schema"], "geofem.gui.analysis_panel.v1")
        self.assertIn("static_plane_strain", contract["analysis_types"])
        self.assertIn("axisymmetric_static", contract["analysis_types"])
        self.assertEqual(contract["analysis_geometries"], ["plane_strain", "axisymmetric"])
        self.assertEqual(contract["deformation_modes"], ["small_deformation", "large_deformation"])
        fields = {field["attribute"]: field["label"] for field in contract["fields"]}
        self.assertEqual(fields["analysis_type"], "解析種別")
        self.assertEqual(fields["analysis_deformation_mode"], "変形モード")
        self.assertEqual(fields["unit_system"], "単位系")
        self.assertEqual(contract["apply_callback"], "apply_analysis_panel")
        self.assertIn("load_selected_input_template", contract["template_callbacks"])

    def test_domain_panel_builders_are_split_from_main_window_layout_code(self) -> None:
        contract = domain_panel_contract()
        self.assertEqual(contract["schema"], "geofem.gui.domain_panels.v1")
        self.assertEqual(contract["builder_count"], 9)
        self.assertTrue({"geometry", "mesh", "materials", "stages", "results", "report"}.issubset(set(contract["panel_keys"])))
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return build_geometry_panel(self, self._domain_panel_qt())", main_window_source)
        self.assertIn("return build_results_panel(self, self._domain_panel_qt())", main_window_source)
        self.assertNotIn("self.stage_detail_type = QComboBox()", main_window_source)
        self.assertNotIn("self.result_component = QComboBox()", main_window_source)

    def test_stage_controller_is_split_from_main_window_operation_handlers(self) -> None:
        contract = stage_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.stage_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 50)
        self.assertIn("apply_stage_detail_tables", contract["methods"])
        self.assertIn("refresh_stage_difference_table", contract["methods"])
        self.assertIn("collect_stage_cumulative_conflicts", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return stage_controller.apply_stage_detail_tables(self, self._stage_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def apply_stage_detail_tables(self) -> None:", main_window_source)
        self.assertNotIn("def collect_stage_cumulative_conflicts(self) -> list[dict[str, Any]]:", main_window_source)

    def test_post_report_controller_is_split_from_main_window_operation_handlers(self) -> None:
        contract = post_report_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.post_report_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 70)
        self.assertIn("load_result_table", contract["methods"])
        self.assertIn("export_scene_pdf", contract["methods"])
        self.assertIn("build_selected_report", contract["methods"])
        self.assertIn("_render_result_table_page", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return post_report_controller.load_result_table(self, self._post_report_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return post_report_controller.build_selected_report(self, self._post_report_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def load_result_table(self, kind: str) -> None:", main_window_source)
        self.assertNotIn("def build_selected_report(self) -> None:", main_window_source)

    def test_post_drawing_helpers_are_split_from_main_window_rendering_logic(self) -> None:
        contract = post_drawing_helper_contract()
        self.assertEqual(contract["schema"], "geofem.gui.post_drawing_helpers.v1")
        self.assertGreaterEqual(contract["method_count"], 20)
        self.assertIn("_draw_result_overlays", contract["methods"])
        self.assertIn("_draw_stage_diff_overlay", contract["methods"])
        self.assertIn("_draw_result_legend", contract["methods"])
        self.assertIn("_contour_color", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return post_drawing_helpers._draw_result_overlays(self, self._post_drawing_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return post_drawing_helpers._draw_result_legend(self, self._post_drawing_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def _draw_result_overlays(self, mesh: Any, scale: float, ox: float, oy: float) -> None:", main_window_source)
        self.assertNotIn("def _contour_color(self, value: float, vmin: float, vmax: float, alpha: int) -> QColor:", main_window_source)

    def test_boundary_load_controller_is_split_from_main_window_input_handlers(self) -> None:
        contract = boundary_load_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.boundary_load_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 40)
        self.assertIn("apply_boundary_conditions_panel", contract["methods"])
        self.assertIn("add_selected_hydro_boundary_condition", contract["methods"])
        self.assertIn("apply_loads_panel", contract["methods"])
        self.assertIn("add_panel_earthquake_load", contract["methods"])
        self.assertIn("apply_axisymmetric_load_preset", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return boundary_load_controller.apply_loads_panel(self, self._boundary_load_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return boundary_load_controller.apply_boundary_conditions_panel(self, self._boundary_load_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def apply_loads_panel(self) -> None:", main_window_source)
        self.assertNotIn("def add_selected_hydro_boundary_condition(self, *_args: Any) -> None:", main_window_source)

    def test_material_controller_is_split_from_main_window_input_handlers(self) -> None:
        contract = material_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.material_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 18)
        self.assertEqual(contract["pure_helper_count"], 8)
        self.assertIn("apply_materials_panel", contract["methods"])
        self.assertIn("add_material_from_library", contract["methods"])
        self.assertIn("estimate_material_constants_from_curve", contract["methods"])
        self.assertIn("_fit_material_curve_global", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return material_controller.apply_materials_panel(self, self._material_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return material_controller._fit_material_curve_global(curve_sets, value_kind)", main_window_source)
        self.assertNotIn("def apply_materials_panel(self) -> None:", main_window_source)
        self.assertNotIn("from scipy.optimize import least_squares", main_window_source)

    def test_mesh_controller_is_split_from_main_window_input_handlers(self) -> None:
        contract = mesh_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.mesh_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 35)
        self.assertIn("mesh_quality", contract["covered_surfaces"])
        self.assertIn("element_library", contract["covered_surfaces"])
        self.assertIn("apply_mesh_panel", contract["methods"])
        self.assertIn("populate_mesh_control_tables", contract["methods"])
        self.assertIn("compare_mesh_quality_improvements_async", contract["methods"])
        self.assertIn("apply_element_library_panel", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return mesh_controller.apply_mesh_panel(self, self._mesh_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return mesh_controller.apply_element_library_panel(self, self._mesh_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def apply_mesh_panel(self) -> None:", main_window_source)
        self.assertNotIn("def populate_mesh_control_tables(self) -> None:", main_window_source)

    def test_mesh_auto_controller_is_split_from_main_window_auto_and_drag_handlers(self) -> None:
        contract = mesh_auto_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.mesh_auto_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 14)
        self.assertIn("auto_mesh_generation", contract["covered_surfaces"])
        self.assertIn("mesh_control_drag", contract["covered_surfaces"])
        self.assertIn("mesh_edge_selection", contract["covered_surfaces"])
        self.assertIn("apply_auto_geometry_mesh_async", contract["methods"])
        self.assertIn("begin_mesh_control_drag", contract["methods"])
        self.assertIn("_selected_mesh_edge_points", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return mesh_auto_controller.apply_auto_geometry_mesh_async(self, self._mesh_auto_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return mesh_auto_controller.begin_mesh_control_drag(self, self._mesh_auto_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def apply_auto_geometry_mesh_async(self) -> None:", main_window_source)
        self.assertNotIn("def begin_mesh_control_drag(self, data: Mapping[str, Any]) -> None:", main_window_source)

    def test_geometry_controller_is_split_from_main_window_cad_handlers(self) -> None:
        contract = geometry_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.geometry_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 60)
        self.assertIn("geometry_tables", contract["covered_surfaces"])
        self.assertIn("cad_dimension_constraints", contract["covered_surfaces"])
        self.assertIn("curve_controls", contract["covered_surfaces"])
        self.assertIn("cad_boolean", contract["covered_surfaces"])
        self.assertIn("apply_geometry_panel", contract["methods"])
        self.assertIn("rebuild_geometry_boolean_graph", contract["methods"])
        self.assertIn("_cad_kernel_constraints_from_geometry", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return geometry_controller.apply_geometry_panel(self, self._geometry_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return geometry_controller.rebuild_geometry_boolean_graph(self, self._geometry_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def apply_geometry_panel(self, *_args: Any, after_change: bool = True) -> bool:", main_window_source)
        self.assertNotIn("def rebuild_geometry_boolean_graph(self) -> None:", main_window_source)

    def test_selection_controller_is_split_from_main_window_selection_and_snap_handlers(self) -> None:
        contract = selection_controller_contract()
        self.assertEqual(contract["schema"], "geofem.gui.selection_controller.v1")
        self.assertGreaterEqual(contract["method_count"], 45)
        self.assertIn("selection_modes", contract["covered_surfaces"])
        self.assertIn("filter_expression_selection", contract["covered_surfaces"])
        self.assertIn("selection_history", contract["covered_surfaces"])
        self.assertIn("snap_helpers", contract["covered_surfaces"])
        self.assertIn("set_selection_mode", contract["methods"])
        self.assertIn("select_by_filter", contract["methods"])
        self.assertIn("save_named_selection", contract["methods"])
        self.assertIn("_snap_model_point", contract["methods"])
        self.assertIn("_selected_model_points", contract["methods"])
        self.assertIn("_nearest_node_id", contract["methods"])
        main_window_source = Path("geofem_app/gui/main_window.py").read_text(encoding="utf-8")
        self.assertIn("return selection_controller.select_by_filter(self, self._selection_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertIn("return selection_controller._snap_model_point(self, self._selection_controller_qt(), *args, **kwargs)", main_window_source)
        self.assertNotIn("def select_by_filter(self, *, kind: str, value: str | None = None, mode: str = \"replace\") -> int:", main_window_source)
        self.assertNotIn("def _snap_model_point(self, x: float, y: float) -> tuple[float, float, str | None]:", main_window_source)

    def test_yaml_panel_contract_covers_solver_and_root_yaml_surfaces(self) -> None:
        contract = yaml_panel_contract()
        self.assertEqual(contract["schema"], "geofem.gui.yaml_panels.v1")
        self.assertEqual(contract["editor_attribute_pattern"], "{key}_editor")
        self.assertEqual(contract["apply_callback"], "apply_yaml_fragment")
        self.assertEqual(contract["root_sync_callback"], "sync_from_yaml")
        self.assertEqual(contract["fragment_expected_types"], ["list", "mapping"])
        self.assertIn("solver", contract["surfaces"])
        self.assertIn("root_yaml", contract["surfaces"])

    def test_model_check_panel_contract_and_summary_are_split_from_main_window(self) -> None:
        contract = model_check_panel_contract()
        self.assertEqual(contract["schema"], "geofem.gui.model_check_panel.v1")
        self.assertEqual(contract["headers"], ["区分", "対象", "内容"])
        self.assertEqual(contract["run_callback"], "run_model_check_async")
        counts = summarize_model_check_issues(
            [
                ("ERROR", "mesh", "bad", {}),
                ("WARN", "material", "soft", {}),
                ("INFO", "solver", "ok", {}),
            ]
        )
        self.assertEqual(counts, {"ERROR": 1, "WARN": 1, "INFO": 1, "TOTAL": 3})

    def test_workflow_panel_specs_are_split_from_main_window(self) -> None:
        groups = workflow_panel_action_groups()
        self.assertEqual([group["title"] for group in groups], ["1. モデル作成", "2. メッシュ分割", "3. ステージ設定", "4-7. 解析・Post・計算書"])
        callbacks = {button["callback"] for group in groups for button in group["buttons"]}
        self.assertIn("run_solver", callbacks)
        self.assertIn("confirm_mesh_generation", callbacks)
        self.assertTrue(all(button["label"] for group in groups for button in group["buttons"]))
        layout = workflow_panel_layout_contract()
        self.assertFalse(layout["scrollable"])
        self.assertEqual(layout["primary_placement"], "model_top_left_wizard")
        self.assertEqual(layout["details_placement"], "workflow_diagnostics_panel")
        self.assertEqual(layout["vertical_compression"], "responsive")
        self.assertEqual(layout["compact_vertical_compression"], "single_row_summary")
        self.assertEqual(layout["expanded_vertical_compression"], "two_row_top_grid")
        self.assertFalse(layout["previous_action_button"])
        self.assertEqual(layout["canonical_navigation"], "workflow_ribbon")
        self.assertFalse(layout["diagnostic_navigation_aliases_visible"])
        self.assertTrue(layout["diagnostics_are_secondary"])
        self.assertGreaterEqual(layout["button_min_height"], 36)
        self.assertGreaterEqual(layout["ribbon_button_min_height"], 28)
        self.assertLessEqual(layout["ribbon_max_height"], 116)
        self.assertEqual(layout["compact_guidance_columns"], ["#", "step", "status", "next_action"])
        self.assertLessEqual(layout["guidance_table_max_height"], 240)
        self.assertTrue(layout["next_action_band"])

    def test_gui_i18n_catalog_covers_shared_gui_surfaces(self) -> None:
        validation = validate_gui_i18n_catalog()
        self.assertTrue(validation["passed"], validation["errors"])
        self.assertIn("gui.command.analysis.run.label", available_gui_message_keys("ja"))
        self.assertEqual(gui_message("gui.command.analysis.run.label", locale="en"), "Run Analysis")
        self.assertEqual(gui_message("workflow.step.report.label", locale="en"), "Report")
        self.assertEqual(gui_message("gui.menu.advanced_settings", locale="ja"), "詳細設定")
        self.assertEqual(gui_message("gui.menu.language", locale="en"), "Language")
        self.assertEqual(gui_message("gui.language.en", locale="ja"), "英語")
        cfg = {"analysis": {"dimension": "2D", "type": "static_plane_strain"}}
        guidance = build_workflow_guidance(cfg, locale="en")
        self.assertEqual(guidance["locale"], "en")
        self.assertEqual(guidance["next_step"]["label"], "Mesh")

    def test_gui_japanese_catalog_has_no_mojibake_artifacts(self) -> None:
        mojibake_markers = ("縺", "繝", "荳", "蜈", "譁", "螳", "諠", "邨", "莨", "謫", "鬆")
        message_values = list(gui_message_catalog("ja").values())
        surface_values = [str(row.get("text", "")) for row in surface_text_catalog(locale="ja")]
        offenders = sorted(
            text
            for text in message_values + surface_values
            if any(marker in text for marker in mojibake_markers)
        )
        self.assertEqual(offenders, [])

    def test_gui_i18n_catalog_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_gui_i18n_catalog(Path(tmp) / "i18n")
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            validation = json.loads(Path(paths["validation"]).read_text(encoding="utf-8"))
            self.assertTrue(validation["passed"])

    def test_gui_surface_text_catalog_covers_panel_status_log_and_report_texts(self) -> None:
        validation = validate_surface_text_catalog()
        self.assertTrue(validation["passed"], validation["errors"])
        self.assertTrue({"button", "form", "log", "menu", "panel", "report", "result", "status", "tab", "tree"}.issubset(set(validation["categories"])))
        self.assertEqual(translate_surface_text("解析実行", locale="en"), "Run")
        self.assertEqual(translate_surface_text("Run", locale="ja"), "解析実行")
        self.assertEqual(translate_surface_text("Analysis", locale="ja", category="menu"), "解析")
        self.assertEqual(translate_surface_text("Analysis", locale="ja", category="panel"), "解析条件")
        self.assertEqual(translate_surface_text("変位ベクトル", locale="en", category="tree"), "Displacement Vectors")
        self.assertEqual(translate_surface_text("Result Display", locale="ja", category="form"), "結果表示調整")
        self.assertEqual(translate_surface_text("変形倍率", locale="en", category="form"), "Deformation Scale")
        self.assertEqual(translate_surface_text("投げ縄選択", locale="en", category="button"), "Lasso Select")
        self.assertEqual(translate_surface_text("軸対称プリセット", locale="en", category="form"), "Axisymmetric Presets")
        self.assertEqual(gui_surface_message("dialog.version.title", locale="en"), "Version Information")
        self.assertEqual(gui_surface_message("status.workspace.updated", locale="en", path="x"), "Workspace dashboard updated: x")
        self.assertEqual(gui_surface_message("log.report.selected", locale="en"), "[GeoFEAS Workflow] Selected report builder")
        rows = surface_text_catalog(locale="en")
        self.assertTrue(any(row["key"] == "group.stage_detail" and row["text"] == "Selected Stage Detail" for row in rows))
        self.assertTrue(any(row["key"] == "status.report.note" and row["category"] == "report" for row in rows))
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_surface_text_catalog(Path(tmp) / "surface")
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            payload = json.loads(Path(paths["validation"]).read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload)

    def test_gui_command_catalog_groups_roles_surfaces_and_shortcuts(self) -> None:
        validation = validate_command_catalog()
        self.assertTrue(validation["passed"], validation["errors"])
        self.assertTrue(validate_command_catalog(locale="en")["passed"])
        hierarchy = command_hierarchy()
        self.assertEqual(hierarchy["schema"], "geofem.gui.command_hierarchy.v1")
        self.assertTrue(hierarchy["features"]["toolbar"])
        self.assertTrue(hierarchy["features"]["context_menu"])
        self.assertTrue(hierarchy["features"]["shortcuts"])
        self.assertIn("analysis.run", hierarchy["toolbar"])
        self.assertIn("model.check", hierarchy["contexts"]["tree"])
        self.assertEqual(hierarchy["shortcuts"]["analysis.run"], "F5")
        self.assertEqual(set(hierarchy["roles"]), {"primary", "confirm", "output", "detail"})
        self.assertTrue(all(hierarchy["roles"][role] for role in hierarchy["roles"]))
        self.assertTrue(any(command["target_panel"] == "report" for command in gui_command_catalog()))
        en_catalog = {command["id"]: command for command in gui_command_catalog(locale="en")}
        self.assertEqual(en_catalog["analysis.run"]["label"], "Run Analysis")

    def test_gui_command_catalog_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_command_catalog(Path(tmp) / "commands")
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertIn("contexts", data)
            self.assertIn("panel", data["contexts"])

    def test_workspace_responsibility_contract_fixes_gui_boundaries(self) -> None:
        validation = validate_workspace_contract()
        self.assertTrue(validation["passed"], validation["errors"])
        contract = workspace_responsibility_contract()
        areas = {area["id"]: area for area in contract["areas"]}
        self.assertEqual(areas["left_tree"]["role"], "navigation")
        self.assertIn("詳細入力フォーム", areas["left_tree"]["disallowed"])
        self.assertIn("解析実行", areas["left_tree"]["disallowed"])
        self.assertEqual(areas["center_workspace"]["role"], "primary")
        self.assertIn("結果図", areas["center_workspace"]["allowed"])
        self.assertEqual(areas["right_auxiliary"]["role"], "auxiliary")
        self.assertIn("巨大フォーム", areas["right_auxiliary"]["disallowed"])
        self.assertIn({"width": 1366, "height": 768}, contract["viewport_regression_targets"])
        menu = menu_bar_contract()
        menu_ids = [row["id"] for row in menu["menus"]]
        self.assertLess(menu_ids.index("operations"), menu_ids.index("advanced_settings"))
        file_menu = next(row for row in menu["menus"] if row["id"] == "file")
        self.assertTrue({"file.save_input_as", "file.import", "file.export"}.issubset(set(file_menu["items"])))
        analysis_menu = next(row for row in menu["menus"] if row["id"] == "analysis")
        self.assertIn("analysis.reset_results", analysis_menu["items"])
        self.assertEqual(menu["primary_command_surfaces"]["analysis.run"], "bottom_primary_actions")
        self.assertIn("menu_bar", menu["secondary_command_surfaces"]["analysis.run"])
        views = workspace_view_contract()
        view_by_id = {row["id"]: row for row in views["views"]}
        self.assertEqual(view_by_id["geometry"]["workspace"], "center_workspace")
        self.assertEqual(view_by_id["geometry"]["heading_fields"], ["work_name", "input_status", "next_action"])
        self.assertTrue(view_by_id["results"]["main_region"])

    def test_workspace_contract_artifact_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_workspace_contract(Path(tmp) / "workspace")
            self.assertTrue(Path(paths["json"]).exists())
            data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertTrue(data["validation"]["passed"])
            self.assertEqual(data["menus"]["schema"], "geofem.gui.menu_bar_contract.v1")

    def test_workflow_guidance_reports_next_required_input_and_groups(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"generator": "rectangle", "nx": 0, "ny": 2, "element_type": "QUAD4"},
            "materials": {},
        }
        guidance = build_workflow_guidance(cfg)
        self.assertFalse(guidance["passed"])
        self.assertEqual(guidance["schema"], "geofem.gui.workflow_guidance.v1")
        self.assertEqual(guidance["next_step"]["id"], "mesh")
        mesh_row = next(row for row in guidance["steps"] if row["id"] == "mesh")
        self.assertIn("mesh.nx", mesh_row["missing_paths"])
        self.assertEqual(guidance["action_groups"]["primary"][2]["target_panel"], "mesh")
        self.assertTrue(guidance["features"]["workflow_navigation"])
        self.assertTrue(any(step["id"] == "report" for step in workflow_steps()))

    def test_workflow_guidance_accepts_explicit_element_types_in_templates(self) -> None:
        cfg = plane_strain_patch_sample("TRI6")
        guidance = build_workflow_guidance(cfg)
        mesh_row = next(row for row in guidance["steps"] if row["id"] == "mesh")
        self.assertTrue(mesh_row["completed"])
        self.assertNotIn("mesh.element_type", mesh_row["missing_paths"])

    def test_workflow_guidance_artifacts_mark_results_and_report_complete(self) -> None:
        cfg = plane_strain_quad4_sample()
        cfg["stages"] = [{"name": "Stage-1", "type": "static"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results"
            result_dir.mkdir()
            (result_dir / "summary.json").write_text("{}", encoding="utf-8")
            (result_dir / "standard_report.html").write_text("<html></html>", encoding="utf-8")
            paths = write_workflow_guidance(cfg, root / "workflow", result_dir=result_dir)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            guidance = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertTrue(guidance["passed"])
            rows = {row["id"]: row for row in guidance["steps"]}
            self.assertTrue(rows["results"]["completed"])
            self.assertTrue(rows["report"]["completed"])
            self.assertEqual(guidance["missing_required_count"], 0)

    def test_input_diagnostics_reports_paths_and_suggestions(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "unit_system": "strange"},
            "mesh": {"generator": "rectangle", "nx": 0, "ny": 1, "element_type": "HEX8", "material": "missing"},
            "materials": {"soil": {"nu": 0.3}},
            "unexpected": True,
        }
        summary = diagnose_input_config(cfg)
        self.assertFalse(summary["passed"])
        paths = {issue["path"] for issue in summary["issues"]}
        self.assertIn("mesh.nx", paths)
        self.assertIn("mesh.element_type", paths)
        self.assertIn("mesh.material", paths)
        self.assertIn("materials.soil.E", paths)
        self.assertTrue(all("message" in issue and "suggestion" in issue for issue in summary["issues"]))

    def test_input_assistance_templates_units_ranges_and_artifacts(self) -> None:
        templates = input_assistance_template_catalog()
        self.assertIn("elastic_soil", {row["id"] for row in templates})
        self.assertIn("fixed_bottom", {row["id"] for row in templates})
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN"},
            "mesh": {"generator": "rectangle", "nx": 1, "ny": 1, "material": "soil"},
            "materials": {"soil": {"model": "mohr_coulomb", "E": 20000.0, "nu": 0.3, "cohesion": 10.0, "friction_angle": 30.0, "gamma": 18.0}},
            "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}],
            "loads": [{"set": "top", "fy": -10.0}],
        }
        summary = build_input_assistance_summary(cfg)
        self.assertEqual(summary["schema"], "geofem.input_assistance.v1")
        self.assertTrue(summary["passed"], summary["diagnostics"])
        self.assertIn("field_units", summary["features"])
        rows = {(row["path"], row["field"]): row for row in summary["guidance_rows"]}
        self.assertEqual(rows[("materials.soil.E", "E")]["unit"], "kPa")
        self.assertIn("0 <= nu < 0.5", rows[("materials.soil.nu", "nu")]["recommended"])
        bad = build_input_assistance_summary({**cfg, "materials": {"soil": {"model": "elastic", "E": -1.0, "nu": 0.7}}})
        self.assertFalse(bad["passed"])
        self.assertGreaterEqual(bad["error_count"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_input_assistance_artifacts(summary, Path(tmp))
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["html"]).exists())

    def test_run_solve_writes_case_diagnostics_and_result_view_index(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.yaml"
            input_path.write_text(_yaml_dump(cfg), encoding="utf-8")
            out = root / "out"
            self.assertEqual(run_solve(input_path, output_dir=str(out)), 0)
            self.assertTrue((out / "input_diagnostics.json").exists())
            self.assertTrue((out / "case_manifest.json").exists())
            self.assertTrue((out / "result_view_index.json").exists())
            self.assertTrue((out.parent / "case_history.jsonl").exists())
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("result_view_index", summary)
            view = json.loads((out / "result_view_index.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(view["stage_count"], 1)
            self.assertIn("displacements", view["stages"][0]["node_tables"])
            for name in (
                "mesh_quality.json",
                "mesh_quality_repairs.csv",
                "material_model_catalog.json",
                "material_inventory.json",
                "analysis_log.json",
                "analysis_log.csv",
                "performance_summary.json",
                "performance_summary.csv",
                "performance_kpi_matrix.json",
                "performance_kpi_matrix.csv",
                "performance_kpi_matrix.html",
                "large_model_operations.json",
                "large_model_operations.csv",
                "large_model_operations.html",
                "large_model_node_index.csv",
                "large_model_element_index.csv",
                "reliability_summary.json",
                "reliability_summary.csv",
                "reliability_summary.html",
                "project_audit_trail.json",
                "project_audit_trail.csv",
                "project_audit_trail.html",
                "input_assistance.json",
                "input_assistance.csv",
                "input_assistance.html",
                "standard_report_data.json",
                "standard_report_sections.csv",
                "standard_report.html",
                "standard_report.pdf",
            ):
                self.assertTrue((out / name).exists(), name)
            self.assertIn("mesh_quality", summary)
            self.assertIn("material_models", summary)
            self.assertIn("analysis_log", summary)
            self.assertIn("performance", summary)
            self.assertIn("performance_kpis", summary)
            self.assertIn("large_model_operations", summary)
            self.assertIn("reliability_summary", summary)
            self.assertIn("standard_report", summary)
            manifest = json.loads((out / "case_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("audit_trail", manifest)
            audit = json.loads((out / "project_audit_trail.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["schema"], "geofem.project_audit_trail.v1")
            self.assertGreaterEqual(audit["counts"]["artifact_count"], 1)
            kpis = json.loads((out / "performance_kpi_matrix.json").read_text(encoding="utf-8"))
            self.assertEqual(kpis["schema"], "geofem.performance_kpi_matrix.v1")
            self.assertTrue(all(kpis["area_coverage"][area] for area in KPI_AREAS))
            report_data = json.loads((out / "standard_report_data.json").read_text(encoding="utf-8"))
            self.assertIn("performance_kpis", report_data)
            self.assertIn("performance_kpi_rows", report_data)
            self.assertIn("assembly_elapsed_seconds", report_data["performance"])
            self.assertIn("linear_solve_elapsed_seconds", report_data["performance"])
            self.assertIn("postprocess_elapsed_seconds", report_data["performance"])
            self.assertIn("dominant_category", report_data["performance"])
            self.assertIn("cache_reuse_count", report_data["performance"])
            self.assertIn("reliability_summary", report_data)
            self.assertIn("reliability_stages", report_data)

    def test_output_reliability_and_commercial_quality_gates_write_artifacts(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.yaml"
            input_path.write_text(_yaml_dump(cfg), encoding="utf-8")
            result_dir = root / "out"
            self.assertEqual(run_solve(input_path, output_dir=str(result_dir)), 0)

            gate = run_output_reliability_gate(result_dir, output_dir=root / "gate")
            self.assertTrue(gate["passed"])
            ids = {row["id"] for row in gate["checks"]}
            self.assertIn("stage_count.match", ids)
            self.assertIn("report.manifest.frozen", ids)
            self.assertIn("performance_kpi.area.warm_solver", ids)
            self.assertIn("large_model_operations.feature.node_search_index", ids)
            self.assertIn("large_model_operations.feature.result_table_virtualization", ids)
            self.assertIn("reliability_summary.feature.input_hash", ids)
            self.assertIn("project_audit_trail.feature.output_artifact_hashes", ids)
            self.assertIn("stage.Stage-1.displacements", ids)
            self.assertTrue((root / "gate" / "output_reliability_gate.json").exists())
            self.assertTrue((root / "gate" / "output_reliability_gate.csv").exists())
            self.assertTrue((root / "gate" / "output_reliability_gate.html").exists())

            quality = run_commercial_quality_check(root / "quality", result_dir=result_dir, run_benchmarks=False)
            self.assertTrue(quality["passed"])
            modules = {row["name"]: row for row in quality["modules"]}
            self.assertEqual(modules["output_reliability"]["status"], "passed")
            self.assertIn(modules["post_report_audit"]["status"], {"passed", "warning"})
            self.assertEqual(modules["standard_benchmarks"]["status"], "skipped")
            self.assertTrue((root / "quality" / "commercial_quality_check.json").exists())

            cli_out = root / "quality_cli"
            self.assertEqual(run_commercial_quality(cli_out, result_dir=str(result_dir), run_benchmarks=False), 0)
            self.assertTrue((cli_out / "commercial_quality_check.html").exists())

    def test_failed_solve_writes_failure_report_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "bad.yaml"
            input_path.write_text("analysis:\n  dimension: 2D\nmesh:\n  generator: rectangle\n  nx: 0\nmaterials: {}\n", encoding="utf-8")
            out = root / "failed"
            with self.assertRaisesRegex(Exception, "input diagnostics failed"):
                run_solve(input_path, output_dir=str(out))
            self.assertTrue((out / "failure_report.json").exists())
            self.assertTrue((out / "failure_diagnostics.json").exists())
            self.assertTrue((out / "failure_diagnostics.csv").exists())
            self.assertTrue((out / "failure_diagnostics.html").exists())
            self.assertTrue((out / "failure_recovery_plan.json").exists())
            self.assertTrue((out / "failure_recovery_plan.csv").exists())
            self.assertTrue((out / "failure_recovery_plan.html").exists())
            failure = json.loads((out / "failure_report.json").read_text(encoding="utf-8"))
            analysis = failure["failure_analysis"]
            self.assertEqual(analysis["primary_category"], "mesh_definition")
            self.assertEqual(analysis["primary_input_path"], "mesh.nx")
            self.assertEqual(analysis["primary_gui_panel"], "mesh")
            recovery = failure["failure_recovery"]
            self.assertGreaterEqual(recovery["action_count"], 1)
            self.assertEqual(recovery["actions"][0]["target_panel"], "mesh")
            self.assertIn("再実行", (out / "failure_report.html").read_text(encoding="utf-8"))
            self.assertIn("復旧候補", (out / "failure_report.html").read_text(encoding="utf-8"))
            manifest = json.loads((out / "case_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertGreaterEqual(manifest["diagnostics"]["error_count"], 1)
            self.assertEqual(manifest["error"]["analysis"]["gui_panel"], "mesh")
            self.assertIn("failure_recovery_plan.json", manifest["error"]["analysis"]["recovery_json"])

    def test_failure_diagnostics_classify_solver_error_categories(self) -> None:
        convergence = classify_failure(ValueError("Stage-2: nonlinear step did not converge, residual=1.0e+03"))
        self.assertEqual(convergence["primary_category"], "convergence_failure")
        self.assertEqual(convergence["primary_stage"], "Stage-2")
        self.assertEqual(convergence["primary_gui_panel"], "solver")
        self.assertIn("増分", convergence["primary_recommended_fix"])

        singular = classify_failure(RuntimeError("Stage-1: direct solver failed: matrix is singular"))
        self.assertEqual(singular["primary_category"], "numerical_failure")
        self.assertEqual(singular["primary_gui_panel"], "solver")
        self.assertIn("特異行列", singular["primary_recommended_fix"])

        material = classify_failure(ValueError("material soil: E must be positive"))
        self.assertEqual(material["primary_category"], "material_parameter")
        self.assertEqual(material["primary_target_id"], "soil")
        self.assertEqual(material["primary_gui_panel"], "materials")

    def test_failure_recovery_plan_uses_logs_mesh_quality_and_plasticity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "analysis_log.json").write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "event_type": "convergence_iteration",
                                "stage": "Stage-2",
                                "iteration": 6,
                                "residual_norm": 50.0,
                                "pressure_residual_norm": 2.0,
                                "constraint_norm": 1.0,
                                "dominant_dof": "node 5 uy",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out / "mesh_quality.json").write_text(
                json.dumps({"summary": {"violation_count": 2}, "repair_candidates": [{"element_id": "E1"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            stage_dir = out / "Stage-2"
            stage_dir.mkdir()
            (stage_dir / "element_stresses.csv").write_text(
                "element,plastic,yield_value\nE1,1,1.01\nE2,0,0.2\n",
                encoding="utf-8",
            )

            analysis = classify_failure(RuntimeError("Stage-2: nonlinear step did not converge, residual=5.0e+01"))
            plan = build_failure_recovery_plan(analysis, output_dir=out)
            ids = {row["id"] for row in plan["actions"]}
            self.assertIn("reduce_increment", ids)
            self.assertIn("dominant_residual_dof", ids)
            self.assertIn("mesh_quality_repairs", ids)
            self.assertIn("plastic_concentration", ids)
            self.assertEqual(plan["actions"][0]["priority"], "HIGH")

            paths = write_failure_recovery_plan(plan, out / "recovery")
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["html"]).exists())

    def test_standard_benchmarks_write_one_command_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench"
            summary = run_standard_benchmark_suite(out)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["case_count"], 17)
            self.assertTrue((out / "standard_benchmark_summary.json").exists())
            self.assertTrue((out / "standard_benchmark_checks.csv").exists())
            self.assertTrue((out / "standard_benchmark_report.html").exists())
            self.assertTrue((out / "standard_benchmark_performance.json").exists())
            self.assertTrue((out / "standard_benchmark_performance.csv").exists())
            self.assertTrue((out / "standard_benchmark_numba_warmup.json").exists())
            self.assertTrue((out / "standard_benchmark_quad8_scaling.csv").exists())
            self.assertTrue((out / "standard_benchmark_real_mesh_scaling.json").exists())
            self.assertTrue((out / "standard_benchmark_real_mesh_scaling.csv").exists())
            self.assertTrue((out / "large_deformation_fast_path_matrix.json").exists())
            self.assertTrue((out / "large_deformation_fast_path_matrix.csv").exists())
            self.assertEqual(summary["performance_regression_count"], 0)
            self.assertIn("numba_warmup", summary)
            self.assertGreaterEqual(float(summary["numba_warmup_elapsed_seconds"]), 0.0)
            self.assertGreaterEqual(float(summary["case_elapsed_seconds_excluding_warmup"]), 0.0)
            self.assertGreaterEqual(float(summary["case_solve_elapsed_seconds_excluding_io"]), 0.0)
            self.assertGreaterEqual(float(summary["case_io_report_elapsed_seconds"]), 0.0)
            perf = json.loads((out / "standard_benchmark_performance.json").read_text(encoding="utf-8"))
            self.assertEqual(perf["quad8_scaling_schema"], "geofem.quad8_scaling_benchmark.v1")
            self.assertEqual(perf["numba_warmup_schema"], "geofem.numba_warmup.v1")
            self.assertGreaterEqual(float(perf["benchmark_numba_warmup_elapsed_seconds"]), 0.0)
            self.assertGreaterEqual(int(perf["benchmark_numba_warmup_kernel_count"]), 1)
            self.assertTrue(all(key in perf["cases"][0] for key in ("stage_elapsed_seconds", "cold_run_elapsed_seconds", "solve_elapsed_seconds_excluding_io", "cold_io_report_elapsed_seconds", "stage_io_report_elapsed_seconds", "run_io_report_elapsed_seconds", "assembly_elapsed_seconds", "linear_solve_elapsed_seconds", "postprocess_elapsed_seconds", "coupled_assembly_elapsed_seconds", "dominant_category", "cache_reuse_count")))
            case_names = {row["case"] for row in perf["cases"]}
            self.assertTrue({"nonlinear_static_von_mises", "riks_mpc_lagrange", "axisymmetric_up_lagrange"} <= case_names)
            riks_perf = next(row for row in perf["cases"] if row["case"] == "riks_mpc_lagrange")
            self.assertGreater(float(riks_perf.get("assembly_elapsed_seconds", 0.0) or 0.0), 0.0)
            self.assertGreater(float(riks_perf.get("linear_solve_elapsed_seconds", 0.0) or 0.0), 0.0)
            self.assertGreaterEqual(float(riks_perf.get("cold_io_report_elapsed_seconds", 0.0) or 0.0), 0.0)
            self.assertTrue(any(float(row.get("assembly_elapsed_seconds", 0.0) or 0.0) >= 0.0 for row in perf["cases"]))
            self.assertGreaterEqual(float(perf["numba_kernel_warmup_elapsed_seconds"]), 0.0)
            self.assertGreaterEqual(float(perf["numba_kernel_warm_elapsed_seconds"]), 0.0)
            rows = perf["quad8_scaling"]
            self.assertGreaterEqual(len(rows), 50)
            self.assertTrue(all(row["warm_elapsed_seconds"] >= 0.0 for row in rows))
            self.assertIn("quad8_scaling_csv", perf["paths"])
            self.assertIn("numba_warmup_json", perf["paths"])
            self.assertIn("large_deformation_fast_path_matrix_json", perf["paths"])
            self.assertIn("real_mesh_scaling_json", perf["paths"])
            self.assertTrue(perf["real_mesh_scaling_passed"])
            self.assertGreaterEqual(int(perf["real_mesh_scaling_max_element_count"]), 2048)
            self.assertEqual({1, 4, 16}, {int(row["element_count"]) for row in rows})
            self.assertIn("QUAD4", {row["element_type"] for row in rows})
            self.assertIn("QUAD8", {row["element_type"] for row in rows})
            self.assertEqual({"FULL", "SRI", "B-BAR"}, {row["integration"] for row in rows if row["element_type"] == "QUAD8"})
            self.assertEqual({"plane_strain", "axisymmetric"}, {row["analysis_type"] for row in rows})
            self.assertIn("elastic", {row["material_model"] for row in rows})
            self.assertIn("j2", {row["material_model"] for row in rows})

    def test_mesh_quality_report_lists_repair_candidates(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [4.0, 0.0], "3": [4.0, 0.2], "4": [0.0, 1.0]},
                "elements": [{"id": "bad", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.3}},
            "checks": {"mesh_quality": {"min_angle_deg": 20.0, "max_aspect_ratio": 3.0}},
        }
        mesh = mesh_from_config(cfg)
        report = evaluate_mesh_quality(mesh, cfg)
        self.assertGreaterEqual(report["summary"]["violation_count"], 1)
        self.assertGreaterEqual(len(report["repair_candidates"]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_mesh_quality_report(mesh, Path(tmp), cfg)
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["repairs_csv"]).exists())

    def test_material_catalog_and_inventory_are_schema_driven(self) -> None:
        catalog = material_model_catalog()
        names = {row["name"] for row in catalog}
        self.assertIn("mohr_coulomb", names)
        schema = material_form_schema("bilinear_liquefaction")
        field_names = {field["name"] for field in schema["fields"]}
        self.assertIn("cyclic_resistance_ratio", field_names)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_material_reports(
                {"soil": {"model": "mohr_coulomb", "E": 10000.0, "nu": 0.3, "cohesion": 5.0, "friction_angle": 30.0}},
                Path(tmp),
            )
            self.assertTrue(Path(paths["catalog_json"]).exists())
            self.assertTrue(Path(paths["inventory_csv"]).exists())

    def test_performance_compare_detects_slowdown(self) -> None:
        current = [{"case": "patch", "elapsed_seconds": 2.0, "total_solver_iterations": 2}]
        baseline = {"cases": [{"case": "patch", "elapsed_seconds": 1.0, "total_solver_iterations": 1}]}
        regressions = compare_performance_cases(current, baseline, max_slowdown=1.5)
        self.assertEqual({row["metric"] for row in regressions}, {"elapsed_seconds", "total_solver_iterations"})

    def test_performance_compare_detects_structure_memory_and_fallback_growth(self) -> None:
        current = [
            {
                "case": "nonlinear",
                "elapsed_seconds": 1.0,
                "total_solver_iterations": 4,
                "max_matrix_nnz": 160,
                "estimated_memory_bytes": 300,
                "fallback_count": 1,
                "sparse_builder_to_csr_count": 2,
            }
        ]
        baseline = {
            "cases": [
                {
                    "case": "nonlinear",
                    "elapsed_seconds": 1.0,
                    "total_solver_iterations": 4,
                    "max_matrix_nnz": 100,
                    "estimated_memory_bytes": 100,
                    "fallback_count": 0,
                    "sparse_builder_to_csr_count": 1,
                }
            ]
        }
        regressions = compare_performance_cases(
            current,
            baseline,
            max_slowdown=1.5,
            max_structure_growth=1.5,
            max_memory_growth=2.0,
        )
        self.assertEqual(
            {row["metric"] for row in regressions},
            {
                "max_matrix_nnz",
                "estimated_memory_bytes",
                "fallback_count",
                "sparse_builder_to_csr_count",
            },
        )

    def test_performance_summary_profiles_categories_cache_and_iterations(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 80.0}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "loads": [{"set": "top", "fy": -1.0}],
            "steps": [{"name": "srm", "type": "srm", "srm": {"factors": [1.0, 2.0], "failure_plastic_ratio": 0.0}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, Path(tmp) / "out")
            summary = build_performance_summary(result, case_name="profile-smoke")
            self.assertEqual(summary["profile"]["schema"], "geofem.performance_profile.v1")
            self.assertIn(
                summary["dominant_category"],
                {"assembly", "nonlinear_iteration", "linear_solve", "postprocess", "coupled_assembly", "srm_trial", "io_report"},
            )
            self.assertGreaterEqual(summary["cache_entry_count"], 1)
            self.assertGreaterEqual(summary["cache_reuse_count"], 1)
            self.assertGreaterEqual(summary["iteration_profile_count"], 2)
            self.assertIn("profile", summary["stages"][0])
            slowest = summary["profile"]["iterations"]["slowest"]
            self.assertIn("tangent_internal_assembly_elapsed_seconds", slowest)
            self.assertIn("reduced_matrix_elapsed_seconds", slowest)
            self.assertIn("linear_solve_elapsed_seconds", slowest)

            paths = write_performance_summary(result, Path(tmp) / "perf", case_name="profile-smoke")
            header = Path(paths["csv"]).read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("dominant_category", header)
            self.assertIn("cache_reuse_count", header)
            self.assertIn("cache reuse", Path(paths["html"]).read_text(encoding="utf-8"))
            gui_summary = _performance_result_summary(Path(paths["csv"]))
            self.assertIn("支配カテゴリ", gui_summary)
            self.assertIn("キャッシュ再利用", gui_summary)

            row = benchmark_case_performance("profile-smoke", "nonlinear", result, summary["elapsed_seconds"], passed=True)
            self.assertGreaterEqual(row["cache_reuse_count"], 1)
            self.assertIn("dominant_stage", row)
            self.assertIn("solve_elapsed_seconds_excluding_io", row)
            self.assertIn("run_io_report_elapsed_seconds", row)

    def test_performance_kpi_matrix_covers_commercial_timing_areas(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg["runtime_measurements"] = {
            "gui_response_ms": 120.0,
            "post_elapsed_seconds": 0.02,
            "report_elapsed_seconds": 0.03,
            "mesh_generation_elapsed_seconds": 0.01,
            "numba_warmup_elapsed_seconds": 0.2,
        }
        cfg["performance_budgets"] = {"gui": {"max_response_ms": 250.0}}
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, Path(tmp) / "out")
            matrix = build_result_performance_kpi_matrix(result)
            self.assertEqual(matrix["schema"], "geofem.performance_kpi_matrix.v1")
            self.assertTrue(all(matrix["area_coverage"][area] for area in KPI_AREAS))
            statuses = {(row["area"], row["metric"]): row["status"] for row in matrix["rows"]}
            self.assertEqual(statuses[("gui", "max_response_ms")], "passed")
            self.assertIn(("solver_profile", "dominant_category_elapsed_seconds"), statuses)
            self.assertIn("solver_profile_breakdown", matrix["features"])
            self.assertGreaterEqual(matrix["measured_count"], 10)

    def test_large_model_operations_index_lod_selection_and_table_virtualization(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg["display"] = {"detail_limit": 1, "vector_limit": 10}
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, Path(tmp) / "out")
            profile = build_large_model_operation_profile(result, result_page_size=1)
            self.assertEqual(profile["schema"], "geofem.large_model_operations.v1")
            self.assertIn("node_search_index", profile["features"])
            self.assertIn("element_search_index", profile["features"])
            self.assertEqual(profile["display_lod"]["mode"], "auto-reduced")
            self.assertGreaterEqual(profile["result_table_virtualization"]["table_count"], 3)
            self.assertTrue(profile["response_time"]["passed"], profile["response_time"]["rows"])
            nodes = query_nodes_by_bbox(result.mesh, -1.0, -1.0, 1.0, 1.0)
            elements = query_elements_by_bbox(result.mesh, -1.0, -1.0, 10.0, 10.0)
            self.assertIn("1", nodes)
            self.assertGreaterEqual(len(elements), 1)
            paths = write_large_model_operation_artifacts(result, Path(tmp) / "large")
            for key in ("json", "csv", "html", "node_index_csv", "element_index_csv"):
                self.assertTrue(Path(paths[key]).exists(), key)

    def test_case_output_comparison_writes_numeric_post_and_report_diffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current" / "results"
            baseline = root / "baseline" / "results"
            for result_dir, displacement, report_text in (
                (current, 0.125, "current report"),
                (baseline, 0.100, "baseline report"),
            ):
                stage = result_dir / "stage_1"
                stage.mkdir(parents=True)
                (result_dir / "summary.json").write_text(
                    json.dumps({"stages": [{"name": "Stage-1", "max_displacement": displacement}]}, ensure_ascii=False),
                    encoding="utf-8",
                )
                (stage / "displacements.csv").write_text(
                    f"node_id,ux,uy\n1,{displacement},0.0\n2,{displacement * 2},0.0\n",
                    encoding="utf-8",
                )
                (stage / "post_view.png").write_bytes((report_text + " image").encode("utf-8"))
                (result_dir / "calculation_report.html").write_text(f"<html>{report_text}</html>", encoding="utf-8")
            out = root / "compare"
            comparison = compare_result_cases(current.parent, baseline.parent, output_dir=out, baseline_label="design")
            self.assertEqual(comparison["schema"], "geofem.case_output_comparison.v1")
            self.assertTrue(comparison["passed"])
            self.assertGreaterEqual(comparison["difference_count"], 3)
            self.assertTrue((out / "case_output_comparison.json").exists())
            self.assertTrue((out / "case_output_comparison.csv").exists())
            self.assertTrue((out / "case_output_comparison.html").exists())
            categories = {row["category"] for row in comparison["rows"]}
            self.assertTrue({"numeric", "csv", "post", "report"}.issubset(categories))
            csv_rows = [row for row in comparison["rows"] if row["category"] == "csv"]
            self.assertTrue(any(row["artifact"] == "stage_1/displacements.csv" and row["status"] == "different" for row in csv_rows))

    def test_reliability_summary_explains_result_trust_factors(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, Path(tmp) / "out")
            summary = build_reliability_summary(result)
            self.assertEqual(summary["schema"], "geofem.reliability_summary.v1")
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["input"]["input_sha256"]), 64)
            self.assertIn("convergence_status", summary["features"])
            self.assertIn("equilibrium_residual_and_boundary_reaction_summary", summary["features"])
            self.assertIn("mesh_quality_summary", summary["features"])
            self.assertGreaterEqual(len(summary["stages"]), 1)
            stage = summary["stages"][0]
            self.assertIn("max_abs_reaction", stage)
            self.assertIn("residual_norm", stage)
            ids = {row["id"] for row in summary["checks"]}
            self.assertIn("input.hash", ids)
            self.assertIn("mesh_quality.passed", ids)

    def test_project_audit_trail_unifies_operations_approvals_and_hashes(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg["stage_diff_approvals"] = {
            "Stage-1:loads:fy": {"status": "承認済", "approver": "qa", "at": "2026-05-23T00:00:00"}
        }
        cfg["stage_diff_approval_history"] = [
            {"approval_key": "Stage-1:loads:fy", "action": "approve", "status": "承認済", "approver": "qa", "at": "2026-05-23T00:00:00"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".geofem_audit_log.jsonl").write_text(
                json.dumps({"time": "2026-05-23T00:00:01", "action": "stage_approve", "target": "Stage-1"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            input_path = root / "input.yaml"
            input_path.write_text(_yaml_dump(cfg), encoding="utf-8")
            out = root / "out"
            self.assertEqual(run_solve(input_path, output_dir=str(out)), 0)
            audit = json.loads((out / "project_audit_trail.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"], audit["checks"])
            self.assertIn("gui_operation_log_tail", audit["features"])
            self.assertIn("stage_approval_history", audit["features"])
            self.assertEqual(audit["counts"]["operation_event_count"], 1)
            self.assertGreaterEqual(audit["counts"]["approval_count"], 2)
            self.assertGreaterEqual(audit["counts"]["artifact_count"], 10)
            self.assertTrue(any(row["kind"] == "stage_approval" for row in audit["events"]))
            self.assertTrue(any(row["role"] == "manifest" for row in audit["artifacts"]))
            rebuilt = build_project_audit_trail(out, project_root=root, input_path=input_path)
            self.assertEqual(rebuilt["schema"], "geofem.project_audit_trail.v1")

    def test_version_info_collects_dependency_licenses_external_tools_and_fonts(self) -> None:
        info = build_version_info(include_gui=False)
        self.assertEqual(info["schema"], "geofem.version_info.v1")
        self.assertIn("dependency_license_inventory", info["features"])
        self.assertIn("external_tool_detection", info["features"])
        self.assertIn("gui_font_inventory", info["features"])
        self.assertIn("update_compatibility_policy", info["features"])
        self.assertEqual(info["product"]["input_config_schema"], "geofem.input_config.v1")
        deps = {row["name"]: row for row in info["dependencies"]}
        self.assertIn("numpy", deps)
        self.assertEqual(deps["numpy"]["status"], "installed")
        self.assertIn("DWG converter", {row["name"] for row in info["external_tools"]})
        self.assertEqual(info["fonts"]["status"], "skipped")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_version_info_artifacts(Path(tmp), payload=info)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], "geofem.version_info.v1")

    def test_update_compatibility_migrates_legacy_input_and_revalidates_artifacts(self) -> None:
        legacy = {
            "schema_version": "geofem.input_config.v0",
            "analysis": {"type": "plane_strain", "unit_system": "m-kN"},
            "mesh": {"generator": "rectangle", "nx": 1, "ny": 1, "type": "QUAD4"},
            "material": {"model": "elastic", "E": 10000.0, "nu": 0.3, "gamma": 18.0},
            "boundary": {"set": "bottom", "ux": 0.0, "uy": 0.0},
            "load": {"set": "top", "fy": -1.0},
            "output_dir": "runs/legacy_case",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "results"
            artifacts.mkdir()
            (artifacts / "summary.json").write_text(json.dumps({"dimension": "2D"}, ensure_ascii=False), encoding="utf-8")
            (artifacts / "case_manifest.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
            report = build_update_compatibility_report(legacy, previous_version="0.0.9", artifact_dir=artifacts)
            self.assertEqual(report["schema"], "geofem.update_compatibility.v1")
            self.assertIn("input_config_schema_migration", report["features"])
            migrated = report["migration"]["migrated_config"]
            self.assertEqual(migrated["schema"], "geofem.input_config.v1")
            self.assertEqual(migrated["analysis"]["dimension"], "2D")
            self.assertEqual(migrated["analysis"]["type"], "static_plane_strain")
            self.assertIn("materials", migrated)
            self.assertEqual(migrated["mesh"]["element_type"], "QUAD4")
            self.assertEqual(migrated["mesh"]["material"], "soil")
            self.assertIn("boundary_conditions", migrated)
            self.assertIn("loads", migrated)
            self.assertEqual(migrated["output"]["directory"], "runs/legacy_case")
            self.assertTrue(report["passed"], report)
            self.assertGreater(report["warning_count"], 0)
            paths = write_update_compatibility_artifacts(legacy, root / "upgrade", previous_version="0.0.9", artifact_dir=artifacts)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            guide = Path(paths["guide"]).read_text(encoding="utf-8")
            self.assertIn("GeoFEM 更新・移行ガイド", guide)
            migrated_yaml = yaml.safe_load(Path(paths["migrated_input"]).read_text(encoding="utf-8"))
            self.assertEqual(migrated_yaml["schema"], "geofem.input_config.v1")

    def test_upgrade_check_cli_writes_migration_artifacts(self) -> None:
        legacy = {
            "analysis": {"type": "plane_strain"},
            "mesh": {"generator": "rectangle", "nx": 1, "ny": 1, "type": "QUAD4"},
            "material": {"model": "elastic", "E": 10000.0, "nu": 0.3},
            "boundary": [{"set": "bottom", "ux": 0.0, "uy": 0.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "legacy.yaml"
            input_path.write_text(_yaml_dump(legacy), encoding="utf-8")
            out = root / "upgrade"
            self.assertEqual(run_upgrade_check(input_path, output_dir=str(out), previous_version="0.0.8"), 0)
            self.assertTrue((out / "update_compatibility.json").exists())
            self.assertTrue((out / "migrated_input.yaml").exists())
            payload = json.loads((out / "update_compatibility.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"], payload)
            self.assertEqual(payload["version"]["status"], "newer_current")

    def test_practical_sample_project_suite_writes_teaching_cases_and_expected_results(self) -> None:
        expected_ids = {
            "tunnel_excavation",
            "retaining_excavation",
            "srm_slope",
            "river_liquefaction",
            "vgflow_seepage",
            "coupled_seepage_geofeas",
        }
        catalog = sample_project_catalog()
        self.assertEqual({row["id"] for row in catalog}, expected_ids)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples"
            paths = write_sample_project_suite(root)
            for path in paths.values():
                self.assertTrue(Path(path).exists())
            manifest = json.loads((root / "sample_project_suite_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "geofem.sample_project_suite.v1")
            self.assertEqual(manifest["case_count"], len(expected_ids))
            for case_id in expected_ids:
                case_dir = root / case_id
                self.assertTrue((case_dir / "input.yaml").exists(), case_id)
                self.assertTrue((case_dir / "README.md").exists(), case_id)
                self.assertTrue((case_dir / "expected_results.json").exists(), case_id)
                self.assertTrue((case_dir / "workflow_checklist.json").exists(), case_id)
                self.assertTrue((case_dir / "project.gfemproj").exists(), case_id)
                cfg = yaml.safe_load((case_dir / "input.yaml").read_text(encoding="utf-8"))
                diagnostics = diagnose_input_config(cfg)
                self.assertEqual(diagnostics["error_count"], 0, (case_id, diagnostics["issues"]))
                expected = json.loads((case_dir / "expected_results.json").read_text(encoding="utf-8"))
                self.assertEqual(expected["schema"], "geofem.sample_project.expected_results.v1")
                self.assertIn("summary.json", expected["expected_artifacts"])
            tunnel_cfg = build_sample_project_config("tunnel_excavation")
            self.assertEqual(tunnel_cfg["stages"][1]["geofeas_workflow"], "tunnel_excavation")
            self.assertAlmostEqual(tunnel_cfg["stages"][1]["stress_release"] + tunnel_cfg["stages"][2]["stress_release"], 1.0)
            coupled_cfg = build_sample_project_config("coupled_seepage_geofeas")
            self.assertEqual(coupled_cfg["analysis"]["type"], "vgflow2d")
            self.assertIn("exchange", coupled_cfg["vgflow2d"])

    def test_sample_projects_cli_writes_selected_practical_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "selected_samples"
            self.assertEqual(run_sample_projects(out, cases=["tunnel_excavation", "vgflow_seepage"]), 0)
            self.assertTrue((out / "sample_project_suite_manifest.json").exists())
            self.assertTrue((out / "tunnel_excavation" / "input.yaml").exists())
            self.assertTrue((out / "vgflow_seepage" / "expected_results.json").exists())
            self.assertFalse((out / "srm_slope").exists())

    def test_workspace_dashboard_lists_recent_runs_artifacts_storage_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            project = new_default_project(root, name="Commercial Workspace")
            input_path = root / "input" / "case.yaml"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(_yaml_dump(plane_strain_quad4_sample()), encoding="utf-8")
            run_dir = root / "runs" / "run_001"
            stage_dir = run_dir / "stage_1"
            stage_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps({"stage_count": 1}, ensure_ascii=False), encoding="utf-8")
            (stage_dir / "displacements.csv").write_text("node_id,ux,uy\n1,0,0\n", encoding="utf-8")
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (root / "reports" / "design_report.pdf").write_bytes(b"%PDF-GeoFEM workspace test")
            manifest = {
                "schema": "geofem.case_manifest.v1",
                "status": "completed",
                "input_file": str(input_path),
                "output_dir": str(run_dir),
                "started_at": "2026-05-23T00:00:00",
                "finished_at": "2026-05-23T00:00:02",
                "elapsed_seconds": 2.0,
                "result": {"stage_count": 1, "warning_count": 0},
            }
            (run_dir / "case_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            (root / "runs" / "case_history.jsonl").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
            project.input_file = str(input_path)
            project.latest_run = str(run_dir)
            save_project(project)

            dashboard = build_workspace_dashboard(root, storage_warning_bytes=1)
            self.assertEqual(dashboard["schema"], WORKSPACE_DASHBOARD_SCHEMA)
            self.assertTrue({"project_dashboard", "recent_analysis_history", "artifact_inventory", "storage_management", "workspace_archive"}.issubset(dashboard["features"]))
            self.assertGreaterEqual(dashboard["counts"]["project_file_count"], 1)
            self.assertGreaterEqual(dashboard["counts"]["run_count"], 1)
            self.assertGreaterEqual(dashboard["counts"]["artifact_count"], 4)
            self.assertTrue(dashboard["storage"]["over_budget"])
            self.assertTrue(any(row["role"] == "report" for row in dashboard["artifacts"]))

            paths = write_workspace_dashboard(root, root / "runs" / "workspace_dashboard", create_archive=True, storage_warning_bytes=1)
            for key in ("json", "runs_csv", "artifacts_csv", "storage_csv", "html", "archive", "archive_manifest"):
                self.assertTrue(Path(paths[key]).exists(), key)
            archive_manifest = json.loads(Path(paths["archive_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(archive_manifest["schema"], WORKSPACE_ARCHIVE_SCHEMA)
            with zipfile.ZipFile(paths["archive"]) as zf:
                names = set(zf.namelist())
            self.assertIn("input/case.yaml", names)
            self.assertIn("workspace_archive_manifest.json", names)
            cli_out = root / "runs" / "workspace_dashboard_cli"
            self.assertEqual(run_workspace_dashboard(root, output_dir=cli_out, create_archive=False, storage_warning_bytes=1), 0)
            self.assertTrue((cli_out / "workspace_dashboard.json").exists())

    def test_organization_customization_profile_writes_templates_and_applies_defaults(self) -> None:
        profile = default_organization_profile("ACME Geotech")
        self.assertEqual(profile["schema"], ORGANIZATION_PROFILE_SCHEMA)
        validation = validate_organization_profile(profile)
        self.assertTrue(validation["passed"], validation)
        catalog = project_template_catalog(profile)
        self.assertIn("geofem_review", {row["id"] for row in catalog})
        self.assertIn("vgflow_seepage", {row["id"] for row in catalog})

        cfg = plane_strain_quad4_sample()
        cfg["analysis"].pop("unit_system", None)
        customized = apply_organization_profile(cfg, profile, template_id="geofem_review")
        self.assertEqual(customized["analysis"]["unit_system"], "m-kN")
        self.assertEqual(customized["organization_profile"]["organization"], "ACME Geotech")
        self.assertEqual(customized["report"]["branding"]["organization"], "ACME Geotech")
        self.assertEqual(customized["report"]["template"]["template_id"], "organization_review_a4")
        self.assertEqual(customized["post"]["style"]["palette"], "organization_safety")
        self.assertEqual(diagnose_input_config(customized)["error_count"], 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_customization_artifacts(root / "custom", profile=profile, template_id="geofem_review")
            for key in ("profile_json", "profile_yaml", "validation", "catalog_json", "catalog_csv", "catalog_html", "customized_sample"):
                self.assertTrue(Path(paths[key]).exists(), key)
            input_path = root / "input.yaml"
            input_path.write_text(_yaml_dump(cfg), encoding="utf-8")
            out = root / "cli_custom"
            applied_output = root / "applied.yaml"
            self.assertEqual(
                run_customization(
                    out,
                    profile_path=Path(paths["profile_yaml"]),
                    input_path=input_path,
                    template_id="geofem_review",
                    applied_output=applied_output,
                ),
                0,
            )
            applied = yaml.safe_load(applied_output.read_text(encoding="utf-8"))
            self.assertEqual(applied["report"]["branding"]["organization"], "ACME Geotech")
            self.assertTrue((out / "project_template_catalog.html").exists())

    def test_startup_check_runs_sample_and_writes_distribution_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "startup"
            report = run_startup_check(out, include_gui=False, run_sample=True)
            self.assertTrue(report["passed"], report)
            self.assertTrue((out / "startup_check.json").exists())
            self.assertTrue((out / "startup_check.csv").exists())
            self.assertTrue((out / "startup_check.html").exists())
            self.assertTrue((out / "startup_environment_diagnostics.json").exists())
            self.assertTrue((out / "startup_environment_diagnostics.csv").exists())
            self.assertTrue((out / "startup_environment_diagnostics.html").exists())
            self.assertTrue((out / "startup_repair_guide.json").exists())
            self.assertTrue((out / "startup_repair_guide.csv").exists())
            self.assertTrue((out / "startup_repair_guide.html").exists())
            self.assertTrue((out / "startup_repair_guide.md").exists())
            self.assertTrue((out / "version_info.json").exists())
            self.assertTrue((out / "version_info.csv").exists())
            self.assertTrue((out / "version_info.html").exists())
            self.assertTrue((out / "startup_support_manifest.json").exists())
            self.assertTrue((out / "startup_support_package.zip").exists())
            self.assertTrue((out / "sample_run" / "summary.json").exists())
            names = {row["name"] for row in report["checks"]}
            self.assertIn("dependency:numpy", names)
            self.assertIn("file:run_gui.bat", names)
            self.assertIn("environment:cpu", names)
            self.assertIn("permission:output_dir_write", names)
            self.assertIn("numba:cache_write", names)
            self.assertIn("font:jp_gui", names)
            self.assertIn("version_info:dependency_licenses", names)
            self.assertIn("version_info:external_tools", names)
            self.assertIn("sample:solve", names)
            self.assertIn("support_artifacts", report)
            environment = json.loads((out / "startup_environment_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(environment["schema"], "geofem.startup_environment_diagnostics.v1")
            self.assertIn("numba_cache_probe", environment["features"])
            self.assertIn("version_license_inventory", environment["features"])
            guide = json.loads((out / "startup_repair_guide.json").read_text(encoding="utf-8"))
            self.assertEqual(guide["schema"], "geofem.startup_repair_guide.v1")
            manifest = json.loads((out / "startup_support_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("privacy", manifest)
            self.assertTrue(manifest["privacy"]["exclude_personal_info"])
            self.assertIn("personal_path_redaction", manifest["privacy"]["features"])
            with zipfile.ZipFile(out / "startup_support_package.zip") as zf:
                names_in_zip = set(zf.namelist())
            self.assertIn("startup_environment_diagnostics.json", names_in_zip)
            self.assertIn("startup_repair_guide.html", names_in_zip)
            self.assertIn("version_info.json", names_in_zip)

    def test_support_package_options_redact_personal_paths_and_exclude_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "support"
            case_dir = root / "UserCase"
            case_dir.mkdir()
            small_log = case_dir / "case.log"
            small_log.write_text(f"path={root}\nuser=link_\n", encoding="utf-8")
            large_json = case_dir / "large_result.json"
            large_json.write_text("x" * 200, encoding="utf-8")
            dump_file = case_dir / "raw.dump"
            dump_file.write_text("dump", encoding="utf-8")
            report = {
                "passed": True,
                "error_count": 0,
                "warning_count": 0,
                "project_root": str(root),
                "environment": {
                    "working_directory": str(root),
                    "python_executable": str(root / "python.exe"),
                },
                "checks": [
                    {
                        "name": "sample:solve",
                        "status": "OK",
                        "detail": str(root),
                        "path": str(small_log),
                    }
                ],
            }
            paths = write_startup_support_artifacts(
                report,
                out,
                {"small": str(small_log), "large": str(large_json), "dump": str(dump_file)},
                options=SupportPackageOptions(
                    exclude_personal_info=True,
                    include_large_results=False,
                    max_file_bytes=120,
                ),
            )
            manifest_text = Path(paths["support_manifest"]).read_text(encoding="utf-8")
            self.assertNotIn(str(root), manifest_text)
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["max_file_bytes"], 120)
            self.assertTrue(manifest["privacy"]["exclude_personal_info"])
            self.assertGreaterEqual(manifest["excluded_large_count"], 1)
            self.assertGreaterEqual(manifest["excluded_by_pattern_count"], 1)
            with zipfile.ZipFile(paths["support_package"]) as zf:
                names = set(zf.namelist())
                self.assertIn("UserCase/case.log", names)
                payload = zf.read("UserCase/case.log").decode("utf-8")
            self.assertNotIn(str(root), payload)
            self.assertIn("<PROJECT_ROOT>", payload)

    def test_api_contracts_validate_public_boundaries_and_write_docs(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        mesh = mesh_from_config(cfg)
        contracts = {row["name"] for row in api_contract_catalog()}
        self.assertIn("input_config", contracts)
        self.assertIn("mesh2d", contracts)
        self.assertTrue(validate_api_contract("input_config", cfg)["passed"])
        self.assertTrue(validate_api_contract("mesh2d", mesh)["passed"])
        invalid = validate_api_contract("input_config", {"analysis": {"dimension": "3D"}})
        self.assertFalse(invalid["passed"])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_api_contract_docs(Path(tmp))
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["markdown"]).exists())

    def test_message_catalog_keeps_diagnostics_contract_stable(self) -> None:
        cfg = {
            "analysis": {"dimension": "3D", "unit_system": "strange"},
            "mesh": {"generator": "rectangle", "nx": 0, "ny": 1, "element_type": "HEX8", "material": "missing"},
            "materials": {"soil": {"nu": 0.3}},
        }
        ja = diagnose_input_config(cfg, locale="ja")
        en = diagnose_input_config(cfg, locale="en")
        ja_keys = [(row["severity"], row["path"]) for row in ja["issues"]]
        en_keys = [(row["severity"], row["path"]) for row in en["issues"]]
        self.assertEqual(ja_keys, en_keys)
        self.assertIn("diagnostics.mesh.missing.message", available_message_keys("ja"))
        self.assertEqual(message("reports.standard.title"), "GeoFEM 2D 標準帳票")
        self.assertFalse(any("\u7e3a" in row["message"] for row in ja["issues"]))

    def test_encoding_policy_detects_non_utf8_and_mojibake_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.md").write_text("# 日本語\nGeoFEM UTF-8\n", encoding="utf-8")
            (root / "bad.md").write_bytes(b"\x82\xa0")
            (root / "mojibake.md").write_text("message: \u7e3a\u52b1", encoding="utf-8")

            summary = audit_text_encoding(root)
            self.assertFalse(summary["passed"])
            checks = {(row["path"], row["check"]) for row in summary["checks"]}
            self.assertIn(("bad.md", "utf8_decode"), checks)
            self.assertIn(("mojibake.md", "mojibake_marker"), checks)

            out = root / "audit"
            written = write_encoding_audit(root, out)
            self.assertTrue((out / "encoding_audit.json").exists())
            self.assertTrue((out / "encoding_audit.csv").exists())
            self.assertTrue((out / "encoding_audit.html").exists())
            self.assertEqual(written["error_count"], 2)

    def test_encoding_policy_cli_and_documentation_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "policy.md").write_text("GeoFEM 日本語 UTF-8\n", encoding="utf-8")
            out = root / "out"
            self.assertEqual(run_encoding_audit(root, out), 0)
            self.assertTrue((out / "encoding_audit.html").exists())
            console = configure_utf8_console()
            self.assertIn("stdout", console)

        policy = Path(__file__).resolve().parents[1] / "docs" / "ENCODING_POLICY_JA.md"
        self.assertTrue(policy.exists())
        text = policy.read_text(encoding="utf-8")
        self.assertIn("UTF-8", text)
        self.assertIn("encoding-audit", text)

    def test_pdf_writer_is_shared_report_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.pdf"
            write_text_pdf(path, ["GeoFEM report", "日本語テキスト"], title="Shared writer")
            data = path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        self.assertIn(b"/Type /Catalog", data)
        self.assertGreater(len(data), 800)

    def test_html_report_helpers_escape_and_link_report_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "reports" / "case.html"
            html = table(["A&B"], [["<raw>"]])
            raw_html = table(["A"], [["<b>ok</b>"]], raw_columns={0})
            self.assertIn("&lt;raw&gt;", html)
            self.assertIn("<b>ok</b>", raw_html)
            self.assertEqual(format_value(1.234567891), "1.2345679")
            self.assertEqual(html_escape("A&B"), "A&amp;B")
            self.assertIn("reports/case.html", rel_link(child, root))
            self.assertIn("class=\"kv\"", kv_table([("key", "value")]))

    def test_low_priority_docs_are_present_and_backlog_has_no_open_items(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in (
            "docs/INSTALL_STARTUP_JA.md",
            "docs/TUTORIAL_BASIC_JA.md",
            "docs/API_CONTRACTS_JA.md",
            "docs/DEVELOPER_GUIDE_JA.md",
        ):
            path = root / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(len(path.read_text(encoding="utf-8")), 500)
        backlog = (root / "GeoFEM_completion_backlog.md").read_text(encoding="utf-8")
        self.assertNotIn("## 優先度低", backlog)
        self.assertIn("未達項目はありません", backlog)

    def test_refactored_seepage_module_preserves_geofeas_verification_facade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vgflow_result.csv"
            path.write_text("time,node_id,pore_pressure,head\n0.0,1,12.5,3.2\n", encoding="utf-8")
            direct = import_seepage_direct(path)
            facade = import_seepage_facade(path)
            self.assertEqual(direct, facade)
            self.assertEqual(direct[0]["source_product"], "VGFlow")
            self.assertAlmostEqual(direct[0]["pore_pressure"], 12.5)

    def test_gui_result_table_routes_are_outside_main_window(self) -> None:
        root = Path("C:/tmp/geofem-run/results")
        stage = root / "Stage-1"
        summary = root / "summary.json"
        self.assertIn("standard_report", known_result_table_kinds())
        self.assertIn("large_model_operations", known_result_table_kinds())
        self.assertEqual(result_table_path("displacements", stage_dir=stage, summary_path=None), stage / "displacements.csv")
        self.assertEqual(result_table_path("standard_report", stage_dir=stage, summary_path=summary), root / "standard_report_sections.csv")
        self.assertEqual(result_table_path("node_search_index", stage_dir=stage, summary_path=summary), root / "large_model_node_index.csv")
        with self.assertRaises(ValueError):
            result_table_path("standard_report", stage_dir=stage, summary_path=None)

    def test_maintainability_audit_finds_large_split_candidates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        summary = audit_maintainability(root, large_line_threshold=1000)
        candidates = {row["path"] for row in summary["split_candidates"]}
        self.assertIn("geofem_app\\gui\\main_window.py", candidates)
        self.assertIn("geofem_app\\fem2d_solver.py", candidates)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_maintainability_audit(summary, Path(tmp))
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["html"]).exists())

    def test_dynamic_helpers_are_split_from_solver_orchestration(self) -> None:
        contract = dynamic_helper_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.dynamic_helpers.v1")
        self.assertGreaterEqual(contract["function_count"], 20)
        self.assertIn("dynamic_profile", contract["covered_surfaces"])
        self.assertIn("time_history_interpolation", contract["covered_surfaces"])
        self.assertIn("rayleigh_damping", contract["covered_surfaces"])
        self.assertIn("mass_regularization", contract["covered_surfaces"])
        self.assertIn("_dynamic_stage_settings", contract["functions"])
        self.assertIn("_time_history_rows", contract["functions"])
        self.assertIn("_dynamic_history_row", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_dynamic import (", solver_source)
        self.assertNotIn("def _dynamic_stage_settings(", solver_source)
        self.assertNotIn("def _dynamic_history_row(", solver_source)

    def test_element_state_output_is_split_from_element_kernels(self) -> None:
        contract = element_state_output_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_state_output.v1")
        self.assertGreaterEqual(contract["function_count"], 4)
        self.assertGreaterEqual(contract["numeric_field_count"], 10)
        self.assertIn("post_material_state", contract["covered_surfaces"])
        self.assertIn("inactive_result_state", contract["covered_surfaces"])
        self.assertIn("_material_state_output", contract["functions"])
        self.assertIn("_inactive_material_state_output", contract["functions"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_state_output import (", elements_source)
        self.assertNotIn("def _material_state_output(", elements_source)
        self.assertNotIn("def _inactive_material_state_output(", elements_source)

    def test_element_interpolation_is_split_from_element_kernels(self) -> None:
        contract = element_interpolation_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_interpolation.v1")
        self.assertGreaterEqual(contract["function_count"], 5)
        self.assertIn("shape_functions", contract["covered_surfaces"])
        self.assertIn("gauss_integration_points", contract["covered_surfaces"])
        self.assertIn("plane_strain_b_matrix", contract["covered_surfaces"])
        self.assertIn("axisymmetric_b_matrix", contract["covered_surfaces"])
        self.assertIn("integration_points", contract["functions"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_interpolation import (", elements_source)
        self.assertNotIn("def shape_functions(", elements_source)
        self.assertNotIn("def integration_points(", elements_source)
        self.assertNotIn("def strain_displacement_matrix(", elements_source)
        self.assertNotIn("def axisymmetric_strain_displacement_matrix(", elements_source)

    def test_element_numba_primitives_are_split_from_element_kernels(self) -> None:
        contract = element_numba_primitives_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_numba_primitives.v1")
        self.assertGreaterEqual(contract["function_count"], 14)
        self.assertIn("quad4_b_matrix_and_jacobian", contract["covered_surfaces"])
        self.assertIn("quad8_shape_gradient_and_b_matrix", contract["covered_surfaces"])
        self.assertIn("quad8_matrix_accumulation", contract["covered_surfaces"])
        self.assertIn("_quad8_gp_full", contract["functions"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_numba_primitives import (", elements_source)
        self.assertNotIn("def _quad4_b_det_numba(", elements_source)
        self.assertNotIn("def _quad4_add_btcb_numba(", elements_source)
        self.assertNotIn("def _quad8_shape_grad_numba(", elements_source)
        self.assertNotIn("def _quad8_add_btstress_numba(", elements_source)

    def test_element_elastic_post_kernels_are_split_from_element_kernels(self) -> None:
        contract = element_elastic_post_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_elastic_post.v1")
        self.assertGreaterEqual(contract["function_count"], 13)
        self.assertIn("quad4_elastic_post", contract["covered_surfaces"])
        self.assertIn("quad8_elastic_post", contract["covered_surfaces"])
        self.assertIn("quad8_elastic_tension_cutoff_post", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        post_source = Path("geofem_app/fem2d_element_post_processing.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_elastic_post import (", elements_source)
        self.assertIn("from .fem2d_element_elastic_post import (", post_source)
        self.assertNotIn("def _quad4_elastic_post_numba(", elements_source)
        self.assertNotIn("def _quad8_elastic_post_numba(", elements_source)
        self.assertNotIn("def _quad8_elastic_tension_post_numba(", elements_source)
        self.assertIn("_quad4_elastic_post_fast(coords, ue, material, initial)", post_source)

    def test_element_elastic_kernels_are_split_from_element_kernels(self) -> None:
        contract = elastic_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_elastic_kernels.v1")
        self.assertGreaterEqual(contract["function_count"], 40)
        self.assertIn("quad4_plane_strain_stiffness_internal_force", contract["covered_surfaces"])
        self.assertIn("quad8_plane_strain_stiffness_internal_force", contract["covered_surfaces"])
        self.assertIn("quad4_pressure_biot_mass", contract["covered_surfaces"])
        self.assertIn("quad8_axisymmetric_stiffness_internal_force", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        kernel_source = Path("geofem_app/fem2d_element_elastic_kernels.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_elastic_kernels import (", elements_source)
        self.assertIn("def _quad4_element_stiffness_numba(", kernel_source)
        self.assertIn("def _quad8_axisymmetric_internal_force_elastic_numba(", kernel_source)
        self.assertNotIn("def _quad4_element_stiffness_numba(", elements_source)
        self.assertNotIn("def _quad8_internal_force_elastic_numba(", elements_source)
        self.assertNotIn("def _quad8_axisymmetric_edge_traction_numba(", elements_source)

    def test_tension_cutoff_kernels_are_split_from_element_kernels(self) -> None:
        contract = tension_cutoff_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_tension_cutoff_kernels.v1")
        self.assertIn("quad8_elastic_tension_cutoff_tangent", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        tension_source = Path("geofem_app/fem2d_element_tension_cutoff_kernels.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_tension_cutoff_kernels import (", elements_source)
        self.assertIn("def _quad8_elastic_tension_tangent_force_numba(", tension_source)
        self.assertNotIn("def _quad8_elastic_tension_tangent_force_numba(", elements_source)
        self.assertNotIn("def _quad8_elastic_tension_tangent_force_fast(", elements_source)

    def test_j2dp_kernels_are_split_from_element_kernels(self) -> None:
        contract = j2dp_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_j2dp_kernels.v1")
        self.assertIn("quad8_j2dp_tangent_internal_force", contract["covered_surfaces"])
        self.assertIn("quad8_axisymmetric_j2dp_post", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        j2dp_source = Path("geofem_app/fem2d_element_j2dp_kernels.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_j2dp_kernels import (", elements_source)
        self.assertIn("def _quad8_j2dp_tangent_force_numba(", j2dp_source)
        self.assertIn("def _quad8_axisymmetric_j2dp_post_numba(", j2dp_source)
        self.assertNotIn("def _quad8_j2dp_tangent_force_numba(", elements_source)
        self.assertNotIn("def _quad4_j2dp_post_numba(", elements_source)
        self.assertNotIn("def _j2dp_stress_tangent_numba(", elements_source)

    def test_mohr_coulomb_kernels_are_split_from_element_kernels(self) -> None:
        contract = mohr_coulomb_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_mohr_coulomb_kernels.v1")
        self.assertIn("quad8_mohr_coulomb_tangent_internal_force", contract["covered_surfaces"])
        self.assertIn("advanced_material_shared_mc_update", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        mc_source = Path("geofem_app/fem2d_element_mohr_coulomb_kernels.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_mohr_coulomb_kernels import (", elements_source)
        self.assertIn("def _quad8_mc_tangent_force_numba(", mc_source)
        self.assertIn("def _quad4_mc_post_update_numba(", mc_source)
        self.assertNotIn("def _quad8_mc_tangent_force_numba(", elements_source)
        self.assertNotIn("def _quad4_mc_post_update_numba(", elements_source)
        self.assertNotIn("def _quad4_mc_stress_tangent_numba(", elements_source)

    def test_advanced_strength_kernels_are_split_from_element_kernels(self) -> None:
        contract = advanced_strength_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_advanced_strength_kernels.v1")
        self.assertIn("quad8_advanced_strength_j2dp_tangent_internal_force", contract["covered_surfaces"])
        self.assertIn("quad8_advanced_strength_mohr_coulomb_post", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        advanced_source = Path("geofem_app/fem2d_element_advanced_strength_kernels.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_advanced_strength_kernels import (", elements_source)
        self.assertIn("def _quad8_advanced_strength_j2dp_tangent_force_numba(", advanced_source)
        self.assertIn("def _quad8_advanced_strength_mc_post_numba(", advanced_source)
        self.assertNotIn("def _quad8_advanced_strength_j2dp_tangent_force_numba(", elements_source)
        self.assertNotIn("def _quad8_advanced_strength_mc_post_numba(", elements_source)
        self.assertNotIn("def _advanced_strength_params_array(", elements_source)

    def test_advanced_elastic_post_kernels_are_split_from_element_kernels(self) -> None:
        contract = advanced_elastic_post_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_advanced_elastic_post.v1")
        self.assertIn("quad8_advanced_elastic_tension_post", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        advanced_source = Path("geofem_app/fem2d_element_advanced_elastic_post.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_advanced_elastic_post import (", elements_source)
        self.assertIn("def _quad8_advanced_elastic_tension_bbar_post_numba(", advanced_source)
        self.assertIn("def _quad4_advanced_elastic_post_fast(", advanced_source)
        self.assertNotIn("def _quad8_advanced_elastic_tension_bbar_post_numba(", elements_source)
        self.assertNotIn("def _quad4_advanced_elastic_post_fast(", elements_source)

    def test_element_fast_path_selection_is_split_from_element_kernels(self) -> None:
        contract = element_fast_path_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_fast_paths.v1")
        self.assertGreaterEqual(contract["function_count"], 30)
        self.assertIn("elastic_post_fast_path_selection", contract["covered_surfaces"])
        self.assertIn("advanced_material_post_fast_path_selection", contract["covered_surfaces"])
        self.assertIn("j2dp_post_fast_path_selection", contract["covered_surfaces"])
        self.assertIn("mohr_coulomb_post_fast_path_selection", contract["covered_surfaces"])
        self.assertIn("plastic_state_arrays", contract["covered_surfaces"])
        self.assertIn("_quad4_post_state_arrays", contract["functions"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_fast_paths import (", elements_source)
        self.assertNotIn("def _quad4_elastic_post_fast_path(", elements_source)
        self.assertNotIn("def _quad8_j2dp_post_fast_path(", elements_source)
        self.assertNotIn("def _quad4_post_state_arrays(", elements_source)
        self.assertNotIn("def _quad8_advanced_state_arrays(", elements_source)

    def test_element_result_rows_are_split_from_element_kernels(self) -> None:
        contract = element_result_row_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_result_rows.v1")
        self.assertGreaterEqual(contract["function_count"], 10)
        self.assertIn("elastic_post_result_rows", contract["covered_surfaces"])
        self.assertIn("advanced_material_result_rows", contract["covered_surfaces"])
        self.assertIn("j2dp_result_rows", contract["covered_surfaces"])
        self.assertIn("mohr_coulomb_result_rows", contract["covered_surfaces"])
        self.assertIn("generic_integration_point_rows", contract["covered_surfaces"])
        self.assertIn("inactive_result_rows", contract["covered_surfaces"])
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_element_result_rows import (", elements_source)
        self.assertNotIn("def _quad4_elastic_post_result_rows(", elements_source)
        self.assertNotIn("def _quad4_advanced_strength_j2dp_post_result_rows(", elements_source)
        self.assertNotIn("def _integration_point_result_row(", elements_source)
        self.assertNotIn("def _inactive_element_result(", elements_source)

    def test_element_post_processing_is_split_from_element_kernels(self) -> None:
        contract = element_post_processing_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_post_processing.v1")
        self.assertIn("integration_point_post_orchestration", contract["covered_surfaces"])
        self.assertIn("fast_post_kernel_dispatch", contract["covered_surfaces"])
        self.assertIn("bbar_fallback_post_processing", contract["covered_surfaces"])
        self.assertEqual(contract["kernel_provider"], "geofem_app.fem2d_elements")
        elements_source = Path("geofem_app/fem2d_elements.py").read_text(encoding="utf-8")
        post_source = Path("geofem_app/fem2d_element_post_processing.py").read_text(encoding="utf-8")
        self.assertIn("fem2d_element_post_processing", elements_source)
        self.assertIn("for element in mesh.elements:", post_source)
        self.assertIn("_quad4_elastic_post_fast(", post_source)
        self.assertNotIn("if _quad4_advanced_elastic_tension_bbar_post_fast_path(element, material):", elements_source)
        self.assertNotIn("_inactive_integration_point_result(element, gp_index", elements_source)

    def test_result_annotations_are_split_from_solver_orchestration(self) -> None:
        contract = result_annotation_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.result_annotations.v1")
        self.assertGreaterEqual(contract["function_count"], 7)
        self.assertIn("matrix_profile_metadata", contract["covered_surfaces"])
        self.assertIn("integration_point_results", contract["covered_surfaces"])
        self.assertIn("liquefaction_state_update", contract["covered_surfaces"])
        self.assertIn("_attach_stage_runtime", contract["functions"])
        self.assertIn("_liquefaction_state_summary", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_result_annotations import (", solver_source)
        self.assertNotIn("def _attach_matrix_profile(", solver_source)
        self.assertNotIn("def _liquefaction_state_summary(", solver_source)

    def test_hydro_helpers_are_split_from_solver_orchestration(self) -> None:
        contract = hydro_helper_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.hydro_helpers.v1")
        self.assertGreaterEqual(contract["function_count"], 18)
        self.assertIn("stage_hydro_merge", contract["covered_surfaces"])
        self.assertIn("seepage_pressure_sync", contract["covered_surfaces"])
        self.assertIn("water_level_pressure_conversion", contract["covered_surfaces"])
        self.assertIn("pressure_constraints", contract["covered_surfaces"])
        self.assertIn("_prepare_stage_hydro", contract["functions"])
        self.assertIn("_collect_pressure_constraints", contract["functions"])
        self.assertIn("_pore_pressure_from_hydro", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_hydro import (", solver_source)
        self.assertNotIn("def _prepare_stage_hydro(", solver_source)
        self.assertNotIn("def _collect_pressure_constraints(", solver_source)

    def test_hydro_iteration_control_is_split_from_solver_orchestration(self) -> None:
        contract = hydro_iteration_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.hydro_iteration.v1")
        self.assertGreaterEqual(contract["function_count"], 6)
        self.assertIn("seepage_fixed_point_stop", contract["covered_surfaces"])
        self.assertIn("seepage_toggle_count", contract["covered_surfaces"])
        self.assertIn("advance_seepage_active_set", contract["functions"])
        self.assertIn("observe_seepage_active_set", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_hydro_iteration import (", solver_source)
        self.assertNotIn("active_signature: tuple[int, int]", solver_source)
        self.assertNotIn("seepage_count\", 0)) == 0 or signature ==", solver_source)

    def test_solver_progress_control_is_split_from_solver_orchestration(self) -> None:
        contract = solver_progress_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.solver_progress.v1")
        self.assertGreaterEqual(contract["function_count"], 12)
        self.assertIn("stage_sequence_normalization", contract["covered_surfaces"])
        self.assertIn("stage_type_and_time", contract["covered_surfaces"])
        self.assertIn("stage_input_merge", contract["covered_surfaces"])
        self.assertIn("stage_solver_config_merge", contract["covered_surfaces"])
        self.assertIn("stage_state_carryover", contract["covered_surfaces"])
        self.assertIn("stage_sequence_from_config", contract["functions"])
        self.assertIn("stage_state_after_result", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_solver_progress import (", solver_source)
        self.assertNotIn("def _stage_type(", solver_source)
        self.assertNotIn("def _stage_time(", solver_source)
        self.assertNotIn('stages_cfg = cfg.get("stages", cfg.get("steps"))', solver_source)

    def test_constraint_helpers_are_split_from_solver_orchestration(self) -> None:
        contract = constraint_helper_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.constraint_helpers.v1")
        self.assertGreaterEqual(contract["function_count"], 4)
        self.assertIn("boundary_condition_dof_mapping", contract["covered_surfaces"])
        self.assertIn("inactive_node_constraints", contract["covered_surfaces"])
        self.assertIn("mpc_penalty_matrix", contract["covered_surfaces"])
        self.assertIn("mpc_violation_postcheck", contract["covered_surfaces"])
        self.assertIn("collect_constraints", contract["functions"])
        self.assertIn("assemble_mpc_penalty", contract["functions"])
        self.assertIn("_add_inactive_node_constraints", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_constraints import (", solver_source)
        self.assertNotIn("def collect_constraints(", solver_source)
        self.assertNotIn("def assemble_mpc_penalty(", solver_source)
        self.assertNotIn("def _add_inactive_node_constraints(", solver_source)

    def test_pressure_assembly_is_split_from_solver_orchestration(self) -> None:
        contract = pressure_assembly_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.pressure_assembly.v1")
        self.assertGreaterEqual(contract["function_count"], 11)
        self.assertIn("plane_strain_pressure_matrices", contract["covered_surfaces"])
        self.assertIn("axisymmetric_pressure_matrices", contract["covered_surfaces"])
        self.assertIn("biot_coupling", contract["covered_surfaces"])
        self.assertIn("pressure_boundary_terms", contract["covered_surfaces"])
        self.assertIn("liquefaction_pressure_terms", contract["covered_surfaces"])
        self.assertIn("assemble_pressure_matrices", contract["functions"])
        self.assertIn("assemble_axisymmetric_pressure_matrices", contract["functions"])
        self.assertIn("assemble_liquefaction_pressure_terms", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_pressure import (", solver_source)
        self.assertNotIn("def assemble_pressure_matrices(", solver_source)
        self.assertNotIn("def assemble_axisymmetric_pressure_matrices(", solver_source)
        self.assertNotIn("def assemble_liquefaction_pressure_terms(", solver_source)

    def test_structural_assembly_is_split_from_solver_orchestration(self) -> None:
        contract = structural_assembly_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.structural_assembly.v1")
        self.assertGreaterEqual(contract["function_count"], 12)
        self.assertIn("linear_stiffness_assembly", contract["covered_surfaces"])
        self.assertIn("consistent_and_lumped_mass", contract["covered_surfaces"])
        self.assertIn("plane_strain_load_vector", contract["covered_surfaces"])
        self.assertIn("axisymmetric_stiffness_assembly", contract["covered_surfaces"])
        self.assertIn("axisymmetric_load_vector", contract["covered_surfaces"])
        self.assertIn("assemble_global_stiffness", contract["functions"])
        self.assertIn("assemble_mass_matrix", contract["functions"])
        self.assertIn("assemble_load_vector", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_structural_assembly import (", solver_source)
        self.assertNotIn("def assemble_global_stiffness(", solver_source)
        self.assertNotIn("def assemble_mass_matrix(", solver_source)
        self.assertNotIn("def assemble_load_vector(", solver_source)

    def test_nonlinear_assembly_is_split_from_solver_orchestration(self) -> None:
        contract = nonlinear_assembly_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.nonlinear_assembly.v1")
        self.assertGreaterEqual(contract["function_count"], 20)
        self.assertIn("plane_strain_algorithmic_tangent", contract["covered_surfaces"])
        self.assertIn("plane_strain_internal_force", contract["covered_surfaces"])
        self.assertIn("axisymmetric_algorithmic_tangent", contract["covered_surfaces"])
        self.assertIn("axisymmetric_internal_force", contract["covered_surfaces"])
        self.assertIn("fast_path_selection", contract["covered_surfaces"])
        self.assertIn("plastic_state_arrays", contract["covered_surfaces"])
        self.assertIn("assemble_algorithmic_tangent_stiffness", contract["functions"])
        self.assertIn("assemble_internal_force", contract["functions"])
        self.assertIn("assemble_axisymmetric_algorithmic_tangent_stiffness", contract["functions"])
        self.assertIn("assemble_axisymmetric_internal_force", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_nonlinear_assembly import (", solver_source)
        self.assertNotIn("def assemble_algorithmic_tangent_stiffness(", solver_source)
        self.assertNotIn("def assemble_internal_force(", solver_source)
        self.assertNotIn("def assemble_axisymmetric_algorithmic_tangent_stiffness(", solver_source)
        self.assertNotIn("def assemble_axisymmetric_internal_force(", solver_source)

    def test_mpc_solver_control_is_directly_owned_by_mpc_module(self) -> None:
        contract = mpc_solver_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.mpc_solver.v1")
        self.assertGreaterEqual(contract["function_count"], 10)
        self.assertIn("exact_elimination_linear_solve", contract["covered_surfaces"])
        self.assertIn("lagrange_multiplier_linear_solve", contract["covered_surfaces"])
        self.assertIn("arc_length_mpc_correction", contract["covered_surfaces"])
        self.assertIn("axisymmetric_up_arc_length_mpc_correction", contract["covered_surfaces"])
        self.assertIn("arc_length_stage_application_plan", contract["covered_surfaces"])
        self.assertIn("stage_application_plan", contract["covered_surfaces"])
        self.assertIn("solve_linear_system_with_mpc_elimination", contract["functions"])
        self.assertIn("solve_linear_system_with_mpc_lagrange", contract["functions"])
        self.assertIn("solve_arc_length_lagrange_correction", contract["functions"])
        self.assertIn("mpc_stage_plan", contract["functions"])
        self.assertIn("mpc_arc_length_stage_plan", contract["functions"])
        solver_source = Path("geofem_app/fem2d_solver.py").read_text(encoding="utf-8")
        self.assertIn("from .fem2d_mpc import (", solver_source)
        self.assertGreaterEqual(solver_source.count("mpc_stage_plan("), 5)
        self.assertNotIn("def solve_linear_system_with_mpc_elimination(", solver_source)
        self.assertNotIn("def solve_linear_system_with_mpc_lagrange(", solver_source)
        self.assertNotIn("def _solve_lagrange_mpc_correction(", solver_source)
        self.assertNotIn("def _solve_arc_length_lagrange_correction(", solver_source)


def _yaml_dump(data: dict) -> str:
    import yaml

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    unittest.main()
