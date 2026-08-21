"""Practical tutorial project packages for GeoFEM and VGFlow2D substitutes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact
from .samples import plane_strain_quad4_sample


SAMPLE_PROJECT_SUITE_SCHEMA = "geofem.sample_project_suite.v1"
SAMPLE_PROJECT_SCHEMA = "geofem.sample_project.v1"
SAMPLE_EXPECTED_SCHEMA = "geofem.sample_project.expected_results.v1"


def sample_project_catalog() -> list[dict[str, Any]]:
    """Return the practical sample projects exposed to CLI/GUI education surfaces."""

    return [
        {
            "id": "tunnel_excavation",
            "title": "NATM tunnel excavation with stress release",
            "domain": "tunnel",
            "solver_route": "GeoFEM 2D",
            "workflow": "tunnel_excavation",
            "level": "intermediate",
            "lesson": "staged excavation, cumulative stress release, GeoFEAS-style Post checks",
        },
        {
            "id": "retaining_excavation",
            "title": "Retaining excavation with staged removal",
            "domain": "retaining_wall",
            "solver_route": "GeoFEM 2D",
            "workflow": "retaining_excavation",
            "level": "intermediate",
            "lesson": "construction stages, excavation target sets, report manifest review",
        },
        {
            "id": "srm_slope",
            "title": "Slope stability strength-reduction method",
            "domain": "slope",
            "solver_route": "GeoFEM 2D",
            "workflow": "srm_slope",
            "level": "advanced",
            "lesson": "SRM factor sweep, integration-point history, safety-factor interpretation",
        },
        {
            "id": "river_liquefaction",
            "title": "River embankment liquefaction substitute",
            "domain": "liquefaction",
            "solver_route": "GeoFEM 2D",
            "workflow": "river_liquefaction_h28",
            "level": "advanced",
            "lesson": "liquefaction metadata, ru/FL history, second-order element guidance",
        },
        {
            "id": "vgflow_seepage",
            "title": "VGFlow2D transient seepage substitute",
            "domain": "seepage",
            "solver_route": "VGFlow2D public substitute",
            "workflow": "vgflow_transient",
            "level": "intermediate",
            "lesson": "known-head boundaries, rainfall, flowline/Post outputs",
        },
        {
            "id": "coupled_seepage_geofeas",
            "title": "VGFlow2D to GeoFEAS handoff substitute",
            "domain": "coupled_seepage_stress",
            "solver_route": "VGFlow2D + GeoFEM handoff",
            "workflow": "vgflow_geofeas_coupling",
            "level": "advanced",
            "lesson": "open exchange files, mesh handoff, water-pressure load review",
        },
    ]


def build_sample_project_config(case_id: str) -> dict[str, Any]:
    """Build a runnable input configuration for a practical tutorial case."""

    case = _normalize_case(case_id)
    if case == "tunnel_excavation":
        cfg = _base_geofem_config(nx=6, ny=3, x_range=[0.0, 18.0], y_range=[0.0, 9.0])
        cfg["sets"] = {"elements": {"top_heading": ["9", "10"], "bench": ["3", "4"], "lining": ["9", "10", "3", "4"]}}
        cfg["stages"] = [
            {"name": "initial-ground", "type": "static", "geofeas_workflow": "tunnel_excavation", "stress_release": 0.0, "loads": [{"set": "top", "ty": -8.0}]},
            {"name": "top-heading", "type": "excavation", "set": "top_heading", "geofeas_workflow": "tunnel_excavation", "stress_release": 0.4, "loads": [{"set": "top", "ty": -8.0}]},
            {"name": "bench", "type": "excavation", "set": "bench", "geofeas_workflow": "tunnel_excavation", "stress_release": 0.6, "loads": [{"set": "top", "ty": -8.0}]},
        ]
        cfg["post"] = {"geofeas_style": True, "relative_displacement": True, "legend_reverse": False}
        return cfg
    if case == "retaining_excavation":
        cfg = _base_geofem_config(nx=4, ny=3, x_range=[0.0, 12.0], y_range=[0.0, 9.0])
        cfg["sets"] = {"elements": {"excavation_lift_1": ["4", "8"], "excavation_lift_2": ["12"], "retained_ground": ["1", "2", "5", "6", "9", "10"]}}
        cfg["stages"] = [
            {"name": "k0-initial", "type": "geostatic", "geofeas_workflow": "retaining_excavation", "k0": 0.5},
            {"name": "excavation-lift-1", "type": "excavation", "set": "excavation_lift_1", "geofeas_workflow": "retaining_excavation", "stress_release": 0.5},
            {"name": "excavation-lift-2", "type": "excavation", "set": "excavation_lift_2", "geofeas_workflow": "retaining_excavation", "stress_release": 0.5},
        ]
        cfg["post"] = {"geofeas_style": True, "value_probe_csv": True}
        return cfg
    if case == "srm_slope":
        cfg = _base_geofem_config(nx=4, ny=2, x_range=[0.0, 16.0], y_range=[0.0, 6.0], material_model="mohr_coulomb")
        cfg["stages"] = [{"name": "srm-factor-sweep", "type": "srm", "geofeas_workflow": "srm_slope", "srm": {"factors": [1.0, 1.2], "failure_plastic_ratio": 0.95}}]
        cfg["post"] = {"geofeas_style": True, "srm_slip_candidates": True}
        return cfg
    if case == "river_liquefaction":
        cfg = _base_geofem_config(nx=3, ny=2, x_range=[0.0, 18.0], y_range=[0.0, 6.0], element_type="QUAD8", material_model="liquefaction")
        cfg["stages"] = [
            {
                "name": "liquefaction-up",
                "type": "consolidation",
                "geofeas_workflow": "river_liquefaction_h28",
                "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 1.0e-7},
            }
        ]
        cfg["post"] = {"geofeas_style": True, "fl_ru_history": True}
        return cfg
    if case == "vgflow_seepage":
        return _vgflow_config(exchange=False)
    if case == "coupled_seepage_geofeas":
        return _vgflow_config(exchange=True)
    raise KeyError(f"unknown sample project case: {case_id}")


def write_sample_project_suite(output_dir: str | Path, *, cases: Sequence[str] | None = None) -> dict[str, str]:
    """Write a suite of practical sample project folders."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = sample_project_catalog()
    selected = _selected_cases(cases)
    project_paths: dict[str, str] = {}
    case_rows = []
    for case_id in selected:
        case_dir = out / case_id
        paths = write_sample_project(case_id, case_dir)
        project_paths[case_id] = str(case_dir)
        spec = _catalog_by_id()[case_id]
        case_rows.append({"id": case_id, "title": spec["title"], "domain": spec["domain"], "solver_route": spec["solver_route"], "workflow": spec["workflow"], "path": str(case_dir)})
    suite = {
        "schema": SAMPLE_PROJECT_SUITE_SCHEMA,
        "case_count": len(selected),
        "features": ["practical_project_folders", "expected_result_contracts", "tutorial_readme", "workflow_checklists", "project_file"],
        "cases": case_rows,
        "project_paths": project_paths,
    }
    paths = {
        "suite_manifest": str(out / "sample_project_suite_manifest.json"),
        "catalog_json": str(out / "sample_project_catalog.json"),
        "catalog_csv": str(out / "sample_project_catalog.csv"),
        "catalog_html": str(out / "sample_project_catalog.html"),
    }
    write_json_artifact(paths["suite_manifest"], suite)
    write_json_artifact(paths["catalog_json"], {"schema": "geofem.sample_project_catalog.v1", "cases": catalog})
    write_dict_rows_csv(paths["catalog_csv"], case_rows, ["id", "title", "domain", "solver_route", "workflow", "path"])
    write_html_artifact(
        paths["catalog_html"],
        html_table_document(
            title="GeoFEM practical sample project catalog",
            lead="Runnable tutorial projects with expected results and workflow checklists.",
            headers=["id", "title", "domain", "solver_route", "workflow", "path"],
            rows=[[row["id"], row["title"], row["domain"], row["solver_route"], row["workflow"], row["path"]] for row in case_rows],
        ),
    )
    return paths


def write_sample_project(case_id: str, output_dir: str | Path) -> dict[str, str]:
    """Write one practical sample project folder."""

    case = _normalize_case(case_id)
    spec = _catalog_by_id()[case]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_sample_project_config(case)
    expected = _expected_results(case, spec, cfg)
    checklist = _workflow_checklist(case, spec)
    project = _project_file_payload(case, spec, cfg)
    paths = {
        "manifest": str(out / "sample_project_manifest.json"),
        "input": str(out / "input.yaml"),
        "expected_results": str(out / "expected_results.json"),
        "workflow_checklist": str(out / "workflow_checklist.json"),
        "readme": str(out / "README.md"),
        "project": str(out / "project.gfemproj"),
    }
    manifest = {
        "schema": SAMPLE_PROJECT_SCHEMA,
        "case": spec,
        "files": paths,
        "run_command": "python -m geofem_app.cli solve input.yaml --out results",
        "expected_results_schema": SAMPLE_EXPECTED_SCHEMA,
    }
    Path(paths["input"]).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json_artifact(paths["expected_results"], expected)
    write_json_artifact(paths["workflow_checklist"], checklist)
    write_json_artifact(paths["project"], project)
    write_json_artifact(paths["manifest"], manifest)
    Path(paths["readme"]).write_text(_readme(case, spec, expected), encoding="utf-8")
    return paths


def _base_geofem_config(
    *,
    nx: int,
    ny: int,
    x_range: list[float],
    y_range: list[float],
    element_type: str = "QUAD4",
    material_model: str = "elastic",
) -> dict[str, Any]:
    cfg = plane_strain_quad4_sample(integration="B-bar")
    cfg["analysis"] = {"dimension": "2D", "type": "static_plane_strain", "unit_system": "m-kN", "profile": "geofeas_public_v5"}
    cfg["mesh"] = {"generator": "rectangle", "x_range": x_range, "y_range": y_range, "nx": nx, "ny": ny, "element_type": element_type, "integration": "B-bar", "material": "soil"}
    cfg["materials"] = {"soil": _material(material_model)}
    cfg["boundary_conditions"] = [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}]
    cfg["loads"] = [{"set": "top", "ty": -5.0}]
    cfg["output"] = {"directory": "results"}
    return cfg


def _material(model: str) -> dict[str, Any]:
    if model == "mohr_coulomb":
        return {"model": "mohr_coulomb", "E": 30000.0, "nu": 0.3, "gamma": 18.0, "cohesion": 12.0, "friction_angle": 32.0, "dilatancy_angle": 0.0}
    if model == "liquefaction":
        return {
            "model": "bilinear_liquefaction",
            "E": 35000.0,
            "G0": 16000.0,
            "gamma_ref": 0.001,
            "nu": 0.3,
            "gamma": 18.0,
            "friction_angle": 30.0,
            "liquefaction": {
                "cyclic_resistance_ratio": 0.22,
                "cyclic_stress_ratio": 0.18,
                "cyclic_stress_method": "gauss_overburden",
                "post_liquefaction_stiffness_ratio": 0.08,
            },
        }
    return {"model": "elastic", "E": 50000.0, "nu": 0.33, "gamma": 18.0}


def _vgflow_config(*, exchange: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "analysis": {"dimension": "2D", "type": "vgflow2d", "mode": "transient", "unit_system": "m-kN"},
        "mesh": {"generator": "rectangle", "x_range": [0.0, 12.0], "y_range": [0.0, 6.0], "nx": 3, "ny": 2, "element_type": "QUAD4", "material": "soil"},
        "materials": {
            "soil": {
                "model": "elastic",
                "E": 1000.0,
                "nu": 0.3,
                "seepage": {
                    "kx": 1.0e-5,
                    "ky": 5.0e-6,
                    "specific_storage": 1.0e-4,
                    "unsaturated_preset": "dam_core",
                },
            }
        },
        "vgflow2d": {
            "mode": "transient",
            "problem_type": "vertical",
            "times": [0.0, 1.0, 2.0],
            "known_head_bcs": [{"set": "left", "head": 4.5}, {"set": "right", "head": 2.0}],
            "rainfall": {"set": "top", "flux": -1.0e-6},
            "seepage_faces": [{"set": "top", "pressure_head": 0.0}],
            "initial_water_level": 3.0,
            "post": {"contour_level_count": 6, "flowline_seed_count": 4, "history_nodes": ["1", "12"]},
        },
        "output": {"directory": "results"},
    }
    if exchange:
        cfg["vgflow2d"]["exchange"] = {"selected_times": [1.0, 2.0], "targets": ["geofeas", "slope_stability"], "water_pressure_load": True}
        cfg["vgflow2d"]["mesh_coupling"] = {"target_element": "QUAD8", "projection": "shape_function"}
    return cfg


def _expected_results(case: str, spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    route = str(spec["solver_route"])
    common_artifacts = ["summary.json"]
    if route.startswith("GeoFEM"):
        common_artifacts.extend(["standard_report.html", "calculation_report_manifest.json", "result_view_index.json"])
    else:
        common_artifacts.extend(["vgflow_node_results.csv", "vgflow_element_results.csv", "vgflow_post_index.html", "vgflow_report_manifest.json"])
    if case == "coupled_seepage_geofeas":
        common_artifacts.extend(["vgflow_exchange_manifest.json", "vgflow_geofeas_stage_handoff.json", "vgflow_coupled_post_comparison.json"])
    return {
        "schema": SAMPLE_EXPECTED_SCHEMA,
        "case_id": case,
        "solver_route": route,
        "expected_artifacts": common_artifacts,
        "acceptance_checks": [
            {"metric": "input_diagnostics.error_count", "operator": "==", "expected": 0},
            {"metric": "summary.dimension", "operator": "==", "expected": "2D"},
            {"metric": "summary.element_count", "operator": ">=", "expected": int(cfg.get("mesh", {}).get("nx", 1))},
            {"metric": "report.readable", "operator": "==", "expected": True},
        ],
        "engineering_notes": [
            "公開情報ベースの代替教材であり、商用製品の公式数値同等性は主張しない。",
            "期待結果は成果物の存在、スキーマ、代表指標、Post/帳票の確認観点を固定する。",
        ],
    }


def _workflow_checklist(case: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    steps = [
        {"order": 1, "panel": "analysis", "action": "解析種別と単位系を確認する", "expected": spec["solver_route"]},
        {"order": 2, "panel": "mesh", "action": "メッシュ分割と要素種別を確認する", "expected": "入力診断エラーなし"},
        {"order": 3, "panel": "materials", "action": "材料パラメータと単位を確認する", "expected": "材料表と帳票へ反映"},
        {"order": 4, "panel": "stages/post", "action": str(spec["lesson"]), "expected": "期待成果物を expected_results.json と照合"},
    ]
    if case.startswith("vgflow") or case == "coupled_seepage_geofeas":
        steps.insert(3, {"order": 4, "panel": "vgflow", "action": "水頭境界、降雨、浸出面、交換時刻を確認する", "expected": "VGFlow公開代替成果物を生成"})
        for index, row in enumerate(steps, start=1):
            row["order"] = index
    return {"schema": "geofem.sample_project.workflow_checklist.v1", "case_id": case, "workflow": spec["workflow"], "steps": steps}


def _project_file_payload(case: str, spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": "GeoFEM project",
        "format_version": 1,
        "sample_project": True,
        "name": spec["title"],
        "dimension": "2D",
        "unit_system": cfg.get("analysis", {}).get("unit_system", "m-kN") if isinstance(cfg.get("analysis", {}), Mapping) else "m-kN",
        "analysis_type": cfg.get("analysis", {}).get("type", "static_plane_strain") if isinstance(cfg.get("analysis", {}), Mapping) else "static_plane_strain",
        "input_file": "input.yaml",
        "latest_run": None,
        "recent_runs": [],
        "run_records": [],
        "model": cfg,
        "tutorial": {"case_id": case, "workflow": spec["workflow"], "expected_results": "expected_results.json"},
    }


def _readme(case: str, spec: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    artifacts = "\n".join(f"- `{name}`" for name in expected.get("expected_artifacts", []))
    return (
        f"# {spec['title']}\n\n"
        f"- ケースID: `{case}`\n"
        f"- 分野: `{spec['domain']}`\n"
        f"- 解析ルート: `{spec['solver_route']}`\n"
        f"- ワークフロー: `{spec['workflow']}`\n\n"
        "## 使い方\n\n"
        "1. `input.yaml` を GUI または CLI で開きます。\n"
        "2. `workflow_checklist.json` の順に入力、解析、Post、帳票を確認します。\n"
        "3. 解析後に `expected_results.json` の成果物と代表チェックを照合します。\n\n"
        "## 期待成果物\n\n"
        f"{artifacts}\n\n"
        "この教材は公開情報ベースの代替仕様です。商用製品の非公開形式や公式数値同等性は検証対象外です。\n"
    )


def _selected_cases(cases: Sequence[str] | None) -> list[str]:
    known = {row["id"] for row in sample_project_catalog()}
    if not cases or any(str(case).lower() == "all" for case in cases):
        return [row["id"] for row in sample_project_catalog()]
    selected = [_normalize_case(case) for case in cases]
    missing = [case for case in selected if case not in known]
    if missing:
        raise KeyError(f"unknown sample project case(s): {', '.join(missing)}")
    return selected


def _catalog_by_id() -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in sample_project_catalog()}


def _normalize_case(case_id: str) -> str:
    return str(case_id).strip().lower().replace("-", "_")


__all__ = [
    "SAMPLE_EXPECTED_SCHEMA",
    "SAMPLE_PROJECT_SCHEMA",
    "SAMPLE_PROJECT_SUITE_SCHEMA",
    "build_sample_project_config",
    "sample_project_catalog",
    "write_sample_project",
    "write_sample_project_suite",
]
