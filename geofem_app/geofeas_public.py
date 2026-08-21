"""Public-information GeoFEAS 2D compatibility helpers.

This module intentionally does not claim binary or numerical equivalence with
commercial GeoFEAS 2D.  It records the public workflow/profile coverage that
GeoFEM can provide from the published product page and operation guidance.
"""

from __future__ import annotations

from datetime import datetime
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fem2d_types import ElasticPlaneStrainMaterial, Mesh2D, SolveResult2D, StageResult2D
from .public_artifacts import write_dict_rows_csv, write_html_artifact, write_json_artifact

PUBLIC_PROFILE = "geofeas_public_v5"

PUBLIC_SOURCES: tuple[dict[str, str], ...] = (
    {
        "label": "FORUM8 GeoFEAS 2D product page",
        "url": "https://www.forum8.co.jp/product/uc1/jiban/geo2d.htm",
    },
    {
        "label": "GeoFEAS 2D operation guidance PDF",
        "url": "https://ftp.forum8.co.jp/forum8lib/jiban/geo2d/geo2d-gui.pdf",
    },
    {
        "label": "FORUM8 GeoFEAS 2D Ver.5 operation guidance movie",
        "url": "https://www.youtube.com/watch?v=C4xlJFnWXZw",
    },
)

LIQUEFACTION_WORKFLOWS = {"river_liquefaction_h19", "river_liquefaction_h28"}
SECOND_ORDER_ELEMENTS = {"TRI6", "QUAD8"}
MOVIE_POST_OUTPUT_ITEMS: tuple[dict[str, str], ...] = (
    {"id": "model_figure", "label": "model figure", "geo_feas_label": "model"},
    {"id": "deformation_figure", "label": "deformation figure", "geo_feas_label": "deformation"},
    {"id": "principal_stress_figure", "label": "principal stress figure", "geo_feas_label": "principal stress"},
    {"id": "principal_strain_figure", "label": "principal strain figure", "geo_feas_label": "principal strain"},
    {"id": "contour_figure", "label": "contour figure", "geo_feas_label": "contour"},
    {"id": "section_force_figure", "label": "section-force figure", "geo_feas_label": "section force"},
    {"id": "numeric_output", "label": "numeric output", "geo_feas_label": "numeric output"},
)
TUNNEL_MOVIE_OPERATION_STEPS: tuple[dict[str, Any], ...] = (
    {"time": "00:15", "tab": "file", "action": "create a new public-profile model from the Tunnel.GF2 guidance scenario", "expected": "new project canvas"},
    {"time": "00:45", "tab": "model", "action": "draw the tunnel inner boundary with regular-polygon/circle registration", "expected": "closed inner tunnel loop"},
    {"time": "01:30", "tab": "model", "action": "draw the outer ground boundary with regular-polygon/circle registration", "expected": "closed outer loop"},
    {"time": "02:30", "tab": "model", "action": "add coordinate-table polygon points for the surrounding ground block", "expected": "outer rectangular block"},
    {"time": "03:00", "tab": "model", "action": "draw horizontal, vertical, and straight auxiliary lines around the tunnel", "expected": "mesh-control line network"},
    {"time": "04:00", "tab": "model", "action": "extend auxiliary lines until they intersect the outer rectangle", "expected": "line network reaches block boundaries"},
    {"time": "04:30", "tab": "model", "action": "generate intersections, select extra lines by rectangle selection, and delete them", "expected": "clean block topology"},
    {"time": "05:00", "tab": "model", "action": "register the first and second ground layer lines from coordinate tables", "expected": "layered ground geometry"},
    {"time": "06:00", "tab": "model", "action": "add horizontal and vertical auxiliary lines around the tunnel and generate intersections", "expected": "layer-aware line topology"},
    {"time": "07:30", "tab": "mesh", "action": "select line groups and enter mesh division counts", "expected": "division counts shown on the model"},
    {"time": "09:15", "tab": "mesh", "action": "assign block property numbers by rectangle selection and solid element selection", "expected": "solid material blocks"},
    {"time": "10:15", "tab": "material", "action": "open material parameter settings and enter public-profile material values", "expected": "material table"},
    {"time": "10:45", "tab": "boundary", "action": "select boundary nodes and assign nodal DOF constraints", "expected": "fixed boundary rows"},
    {"time": "11:30", "tab": "structural", "action": "assign beam element property numbers and material values to selected lines", "expected": "support/beam lines"},
    {"time": "13:30", "tab": "stage", "action": "create an upper-half excavation stage with stress release", "expected": "excavation stage and release metadata"},
    {"time": "16:00", "tab": "stage", "action": "create a lower-half excavation stage with stress release", "expected": "cumulative excavation release metadata"},
    {"time": "17:15", "tab": "solve", "action": "run all stages after selecting output folder and base file name", "expected": "solver progress and stage outputs"},
    {"time": "18:00", "tab": "post", "action": "confirm model, deformation, principal stress, principal strain, contour, section-force, and numeric output views", "expected": "post output condition set"},
    {"time": "18:45", "tab": "post", "action": "save output conditions after output and support explicit overwrite", "expected": "output-condition manifest"},
)

WORKFLOW_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "tunnel_excavation",
        "title": "Tunnel excavation and stress release",
        "stage_pattern": ["initial ground", "excavation", "support", "post"],
        "expected_outputs": ["model", "mesh", "stage log", "deformation", "stress contour", "report manifest"],
    },
    {
        "id": "retaining_excavation",
        "title": "Retaining excavation with construction stages",
        "stage_pattern": ["initial ground", "excavation", "support activation", "post"],
        "expected_outputs": ["stage difference", "active element table", "water pressure note", "case comparison"],
    },
    {
        "id": "river_liquefaction_h19",
        "title": "River levee liquefaction workflow H19",
        "stage_pattern": ["before liquefaction", "during liquefaction", "post-liquefaction volume compression"],
        "expected_outputs": ["FL history", "ru history", "liquefaction post", "numeric value check"],
    },
    {
        "id": "river_liquefaction_h28",
        "title": "River levee liquefaction workflow H28",
        "stage_pattern": ["before liquefaction", "during liquefaction", "post-liquefaction volume compression"],
        "expected_outputs": ["FL history", "ru history", "liquefaction post", "numeric value check"],
    },
    {
        "id": "seepage_pressure",
        "title": "Seepage result import and water pressure load",
        "stage_pattern": ["external head import", "water pressure conversion", "mechanical solve"],
        "expected_outputs": ["pore_pressure.csv", "hydro sync log", "external roundtrip diagnosis"],
    },
    {
        "id": "axisymmetric",
        "title": "Axisymmetric 2D analysis",
        "stage_pattern": ["axisymmetric model", "axisymmetric solve", "axisymmetric post"],
        "expected_outputs": ["sigma_z", "axisymmetric legend note", "report manifest"],
    },
    {
        "id": "srm_slope",
        "title": "Strength reduction slope stability",
        "stage_pattern": ["baseline", "strength reduction trials", "slip candidate post"],
        "expected_outputs": ["factor of safety", "trial table", "integration point history", "SRM post"],
    },
)


def public_profile_enabled(cfg: Mapping[str, Any] | None) -> bool:
    """Return true when the public GeoFEAS profile is requested or implied."""

    if not isinstance(cfg, Mapping):
        return False
    analysis = _mapping(cfg.get("analysis", {}))
    if str(analysis.get("profile", "")).strip().lower() == PUBLIC_PROFILE:
        return True
    post = _mapping(cfg.get("post", cfg.get("postprocess", {})))
    if bool(post.get("geofeas_style", False)):
        return True
    for stage in _stages_from_config(cfg):
        if isinstance(stage, Mapping) and stage.get("geofeas_workflow"):
            return True
    for raw in _mapping(cfg.get("materials", cfg.get("material", {}))).values():
        if isinstance(raw, Mapping) and isinstance(raw.get("liquefaction"), Mapping):
            if raw["liquefaction"].get("cyclic_stress_method"):
                return True
    return False


def workflow_catalog() -> list[dict[str, Any]]:
    return [dict(row) for row in WORKFLOW_CATALOG]


def public_movie_tunnel_operation_steps() -> list[dict[str, Any]]:
    """Return the video-observed tunnel operation steps as an auditable surrogate."""

    return [dict(row) for row in TUNNEL_MOVIE_OPERATION_STEPS]


def public_workflow_operation_log(workflow: str) -> list[dict[str, Any]]:
    """Return a reproducible public-guidance operation log template."""

    if workflow == "tunnel_excavation":
        return [
            {
                "step": index,
                "source": "FORUM8 operation guidance movie",
                "time": str(item.get("time", "")),
                "tab": str(item.get("tab", "")),
                "action": str(item.get("action", "")),
                "expected": str(item.get("expected", "")),
            }
            for index, item in enumerate(TUNNEL_MOVIE_OPERATION_STEPS, start=1)
        ]
    catalog = {row["id"]: row for row in WORKFLOW_CATALOG}
    item = catalog.get(workflow, {"id": workflow or "generic", "stage_pattern": ["model", "mesh", "solve", "post"], "expected_outputs": []})
    pattern = list(item.get("stage_pattern", []))
    outputs = list(item.get("expected_outputs", []))
    log: list[dict[str, Any]] = [
        {"step": 1, "tab": "model", "action": "create block model and assign materials", "expected": "closed CAD/mesh blocks"},
        {"step": 2, "tab": "mesh", "action": "generate and check triangular/quadrilateral mesh", "expected": "mesh quality diagnostics"},
    ]
    for offset, label in enumerate(pattern, start=3):
        log.append({"step": offset, "tab": "stage", "action": f"apply public workflow stage: {label}", "expected": "stage diff and audit log"})
    log.extend(
        [
            {"step": len(log) + 1, "tab": "solve", "action": "run GeoFEM 2D solver", "expected": "stage output directory"},
            {"step": len(log) + 1, "tab": "post", "action": "render public-profile post views", "expected": ", ".join(outputs) or "model/deformation/contour"},
            {"step": len(log) + 1, "tab": "report", "action": "write manifest-backed report", "expected": "calculation_report_manifest.json"},
        ]
    )
    return log


def annotate_public_stage_result(
    stage: StageResult2D,
    cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    stage_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach public GeoFEAS profile metadata to a solved stage."""

    workflow = str(stage_cfg.get("geofeas_workflow", "") or "").strip()
    if not workflow and not public_profile_enabled(cfg):
        return {}
    if not workflow:
        workflow = _infer_workflow(stage, cfg, stage_cfg, materials)
    post = _mapping(cfg.get("post", cfg.get("postprocess", {})))
    history = list(stage_history or [])
    metadata: dict[str, Any] = {
        "schema": "geofem.geofeas_public_stage.v1",
        "profile": PUBLIC_PROFILE,
        "workflow": workflow or "generic",
        "sources": [dict(row) for row in PUBLIC_SOURCES],
        "operation_log": public_workflow_operation_log(workflow),
        "expected_outputs": _expected_outputs(workflow),
        "mesh": _mesh_public_diagnostics(mesh),
        "post": _post_public_metadata(post),
    }
    stress = _stress_release_metadata(stage_cfg, history)
    if stress:
        metadata["stress_release"] = stress
    liquefaction = _liquefaction_public_metadata(stage, cfg, stage_cfg, mesh, materials, workflow)
    if liquefaction:
        metadata["liquefaction"] = liquefaction
    stage.solver_info["geofeas_public"] = metadata
    return metadata


def write_stage_public_profile(stage: StageResult2D) -> Path | None:
    """Write operation log and expected-output metadata for a stage."""

    metadata = stage.solver_info.get("geofeas_public")
    if not isinstance(metadata, Mapping) or stage.output_dir is None:
        return None
    path = Path(stage.output_dir) / "geofeas_public_operation_log.json"
    payload = {
        "schema": "geofem.geofeas_public_operation_log.v1",
        "stage": stage.name,
        "profile": metadata.get("profile", PUBLIC_PROFILE),
        "workflow": metadata.get("workflow", "generic"),
        "operation_log": metadata.get("operation_log", []),
        "expected_outputs": metadata.get("expected_outputs", []),
        "diagnostics": {key: value for key, value in metadata.items() if key not in {"operation_log", "expected_outputs", "sources"}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def public_output_conditions(result: SolveResult2D) -> dict[str, Any]:
    """Return a GeoFEAS-style open substitute for Post output-condition files."""

    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    if not public_profile_enabled(cfg) and not any(isinstance(stage.solver_info.get("geofeas_public"), Mapping) for stage in result.stages):
        return {}
    summary = public_profile_summary(result)
    stages: list[dict[str, Any]] = []
    for index, stage in enumerate(result.stages, start=1):
        public = stage.solver_info.get("geofeas_public", {})
        if not isinstance(public, Mapping):
            public = {}
        post = public.get("post", {})
        if not isinstance(post, Mapping):
            post = {}
        stages.append(
            {
                "index": index,
                "stage": stage.name,
                "workflow": str(public.get("workflow", "")),
                "views": list(post.get("views", [])) or [item["id"] for item in MOVIE_POST_OUTPUT_ITEMS],
                "relative_displacement": bool(post.get("relative_displacement", False)),
                "files": _stage_output_condition_files(stage),
            }
        )
    return {
        "schema": "geofem.geofeas_public_output_conditions.v1",
        "profile": PUBLIC_PROFILE,
        "source_basis": [dict(row) for row in PUBLIC_SOURCES],
        "movie_observed_items": [dict(row) for row in MOVIE_POST_OUTPUT_ITEMS],
        "stages": stages,
        "workflows_used": summary.get("workflows_used", []),
        "save_behavior": {
            "open_substitute": True,
            "format": "JSON",
            "supports_explicit_overwrite": True,
            "commercial_oss_roundtrip": False,
            "blocked_reason": "GeoFEAS *.oss output-condition format is not public.",
        },
    }


def write_public_output_conditions(result: SolveResult2D, path: str | Path) -> Path | None:
    payload = public_output_conditions(result)
    if not payload:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def public_profile_run_warnings(
    cfg: Mapping[str, Any],
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    stages: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return non-fatal warnings for public-profile diagnostics."""

    if not public_profile_enabled(cfg):
        return []
    warnings: list[str] = []
    releases: dict[str, float] = {}
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        release = _normalized_release(stage.get("stress_release", stage.get("stress_release_ratio", stage.get("release_rate"))))
        if release is None:
            continue
        key = str(stage.get("geofeas_workflow", stage.get("set", "global")) or "global")
        releases[key] = releases.get(key, 0.0) + release
    for key, total in sorted(releases.items()):
        if not math.isclose(total, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9):
            warnings.append(f"GeoFEAS public profile: stress_release for {key} totals {total:.6g}, expected 1.0")
    if _has_liquefaction_material(materials) and not _mesh_has_second_order(mesh):
        warnings.append("GeoFEAS public profile: liquefaction guidance expects second-order elements; current mesh uses first-order elements")
    return warnings


def public_profile_summary(result: SolveResult2D) -> dict[str, Any]:
    """Build run-level public-profile metadata from stage annotations."""

    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    stage_items = [stage.solver_info.get("geofeas_public") for stage in result.stages if isinstance(stage.solver_info.get("geofeas_public"), Mapping)]
    if not stage_items and not public_profile_enabled(cfg):
        return {}
    workflows = sorted({str(item.get("workflow", "generic")) for item in stage_items if isinstance(item, Mapping)})
    matrix = build_public_compatibility_matrix(result=result)
    public_rows = [row for row in matrix if row["public_status"] != "blocked_proprietary"]
    blocked_rows = [row for row in matrix if row["public_status"] == "blocked_proprietary"]
    return {
        "schema": "geofem.geofeas_public_profile.v1",
        "profile": PUBLIC_PROFILE,
        "enabled": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [dict(row) for row in PUBLIC_SOURCES],
        "workflows_used": workflows,
        "workflow_count": len(workflows),
        "stage_diagnostics": stage_items,
        "implemented_public_count": len(public_rows),
        "blocked_proprietary_count": len(blocked_rows),
        "compatibility_matrix": matrix,
        "movie_tunnel_operation_steps": public_movie_tunnel_operation_steps() if "tunnel_excavation" in workflows else [],
        "post_output_condition_items": [dict(row) for row in MOVIE_POST_OUTPUT_ITEMS],
        "status": "public_alternative_available",
        "note": "Public workflow coverage is implemented as a GeoFEM substitute; proprietary GeoFEAS data equivalence remains outside scope.",
    }


def write_public_profile_summary(result: SolveResult2D, path: str | Path) -> Path | None:
    summary = public_profile_summary(result)
    if not summary:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


def build_public_compatibility_matrix(*, result: SolveResult2D | None = None, cases: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, str]]:
    """Return the public-info compatibility matrix, including remaining gaps."""

    workflows: set[str] = set()
    if result is not None:
        for stage in result.stages:
            meta = stage.solver_info.get("geofeas_public", {})
            if isinstance(meta, Mapping):
                workflows.add(str(meta.get("workflow", "")))
    if cases:
        for case in cases:
            for tag in _ensure_list(case.get("tags", [])):
                workflows.add(str(tag))
            for workflow in _case_workflows(case):
                workflows.add(workflow)
    implemented = "implemented_public"
    covered = "covered_by_existing_regression"
    blocked = "blocked_proprietary"
    return [
        _matrix_row("P0-1", "P0", "Public operation-guidance workflow templates and replay verification", implemented, "geofeas_public_operation_log.json, workflow catalog, movie-observed tunnel step log, and tools/verify_geofeas_workflow_log.py", "Exact commercial screen transition replay remains unverified"),
        _matrix_row("P0-2", "P0", "Synthetic numeric benchmark suite and reference-package comparator", covered, "benchmark_summary.json, tolerance CSV, HTML report, tools/compare_geofeas_reference_package.py", "Official GeoFEAS sample/result tolerances are unavailable"),
        _matrix_row("P0-3", "P0", "CAD block closure and white-space diagnostics", implemented, "CAD Boolean/auto block diagnostics connected to public matrix", "DWG/SXF product-version fidelity requires commercial samples"),
        _matrix_row("P0-4", "P0", "Mesh division and second-order diagnostics", implemented, "mesh public diagnostics and liquefaction second-order warning", "GeoFEAS mesher exact subdivision sequence is not public"),
        _matrix_row("P0-5", "P0", "Movie-observed tunnel GUI command sequence", implemented, "regular polygon, auxiliary lines, intersection generation, rectangle deletion, division counts, material/support/stage/solve/post steps in operation log", "Pixel-level menu timing and command enablement remain unverified"),
        _matrix_row("P1-5", "P1", "Stage workflow and cumulative stress release", implemented, "steps[].geofeas_workflow and stress_release cumulative metadata", "Exact GeoFEAS dialog warnings need product reference captures"),
        _matrix_row("P1-6", "P1", "Material/liquefaction public substitute model and audit", implemented, "cyclic_stress_method, three-stage liquefaction metadata, and tools/audit_geofeas_material_profile.py", "Official internal constitutive constants/formula variants remain proprietary"),
        _matrix_row("P1-7", "P1", "Liquefaction FL/ru post history", covered, "liquefaction_state.csv, liquefaction_history.csv, liquefaction_ru_fl.svg", "Official FL/RL/N-value result parity is unavailable"),
        _matrix_row("P1-8", "P1", "SRM slope public benchmark", covered, "factor of safety, trials, integration point history", "Official slip-surface search comparison requires product data"),
        _matrix_row("P1-9", "P1", "Structural and joint element post", covered, "interface/structural benchmark cases", "Commercial element library exact parity is unverified"),
        _matrix_row("P1-10", "P1", "Seepage and water-pressure linkage", implemented, "seepage_csv, shared hydro exchange engine, water level conversion, hydro sync metadata", "VGFlow/GeoFEAS native binary exchange remains unavailable"),
        _matrix_row("P1-11", "P1", "Axisymmetric public workflow", covered, "axisymmetric benchmark and sigma_z report metadata", "GeoFEAS axisymmetric UI flow exactness requires product captures"),
        _matrix_row("P2-12", "P2", "Post figures, legends, numeric value checks, and output audit", implemented, "model/deformed/vector/contour/distribution style metadata, SVG report checks, and tools/audit_geofeas_post_report.py", "Pixel-equivalent commercial Post views are unverified"),
        _matrix_row("P2-13", "P2", "PDF/HTML report manifest and audit", covered, "calculation_report_manifest.json with frozen hashes and Post/report audit artifacts", "Commercial report template/layout parity needs official examples"),
        _matrix_row("P2-14", "P2", "Open external format adapters, package inventory, and product-version audit", implemented, "DXF/SXF/P21/GF1/open CSV diagnostics, seepage roundtrip hooks, tools/diagnose_geofeas_package.py, and tools/audit_external_product_versions.py", "Non-public binary GF2/DWG converter parity is blocked"),
        _matrix_row("P2-15", "P2", "Model check and repair candidate diagnostics", implemented, "mesh/CAD/stage/post diagnostics surfaced in GUI and reports", "Exact GeoFEAS error texts and timing remain unverified"),
        _matrix_row("P2-16", "P2", "Post output-condition open substitute", implemented, "geofeas_public_output_conditions.json records movie-observed output items and overwrite/save semantics", "Commercial *.oss binary roundtrip remains blocked"),
        _matrix_row("X-1", "blocked", "Official GeoFEAS numerical equivalence", blocked, "Reference-package comparator is available, but no official samples are bundled", "Requires official *.GF2/*.sta/*.oss or published result tables"),
        _matrix_row("X-2", "blocked", "Private binary roundtrip", blocked, "Package inventory reports blocked/private files and open substitutes, but does not decode private formats", "Requires proprietary format specifications or converter validation data"),
        _matrix_row("X-3", "blocked", "Commercial GUI pixel/wording identity", blocked, "Public workflow replay verifier is available, but pixel/state/wording identity is not asserted", "Requires licensed product captures and acceptance tolerances"),
        _matrix_row("X-4", "blocked", "External product exact version parity", blocked, "External product version audit is available, but exact native exchange parity is not asserted", "Requires VGFlow/GeoFEAS/UWLC/UC-1 product files across supported versions and accepted attribute-preservation tolerances"),
    ]


def write_public_compatibility_matrix(
    output_dir: str | Path,
    *,
    result: SolveResult2D | None = None,
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write JSON/CSV/HTML compatibility matrix artifacts."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = build_public_compatibility_matrix(result=result, cases=cases)
    summary = {
        "schema": "geofem.geofeas_public_compatibility_matrix.v1",
        "profile": PUBLIC_PROFILE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [dict(row) for row in PUBLIC_SOURCES],
        "row_count": len(rows),
        "public_implemented_count": sum(1 for row in rows if row["public_status"] != "blocked_proprietary"),
        "blocked_proprietary_count": sum(1 for row in rows if row["public_status"] == "blocked_proprietary"),
        "passed": all(row["public_status"] != "needs_public_implementation" for row in rows),
        "rows": rows,
    }
    json_path = root / "geofeas_public_compatibility_matrix.json"
    csv_path = root / "geofeas_public_compatibility_matrix.csv"
    html_path = root / "geofeas_public_compatibility_matrix.html"
    write_json_artifact(json_path, summary)
    write_dict_rows_csv(csv_path, rows, ["id", "priority", "title", "public_status", "evidence", "remaining_gap", "source_basis"])
    write_html_artifact(html_path, _matrix_html(summary))
    return {**summary, "json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stages_from_config(cfg: Mapping[str, Any]) -> list[Any]:
    raw = cfg.get("stages", cfg.get("steps", []))
    return list(raw) if isinstance(raw, list) else []


def _infer_workflow(
    stage: StageResult2D,
    cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> str:
    stype = str(stage_cfg.get("type", "")).lower().strip()
    analysis = _mapping(cfg.get("analysis", {}))
    if str(analysis.get("type", "")).lower().strip().startswith("axisymmetric") or stage.solver_info.get("geometry") == "axisymmetric":
        return "axisymmetric"
    if stype in {"srm", "safety_factor"}:
        return "srm_slope"
    if stype in {"excavation", "death", "deactivate"}:
        return "retaining_excavation"
    if _has_liquefaction_material(materials):
        return "river_liquefaction_h28"
    if _mapping(stage_cfg.get("hydro", {})).get("seepage_csv"):
        return "seepage_pressure"
    return "generic"


def _expected_outputs(workflow: str) -> list[str]:
    for row in WORKFLOW_CATALOG:
        if row["id"] == workflow:
            return list(row.get("expected_outputs", []))
    return ["model", "mesh", "deformation", "contour", "numeric value check", "report manifest"]


def _post_public_metadata(post: Mapping[str, Any]) -> dict[str, Any]:
    enabled = bool(post.get("geofeas_style", False))
    return {
        "geofeas_style": enabled,
        "views": [item["id"] for item in MOVIE_POST_OUTPUT_ITEMS] if enabled else [],
        "view_substitutes": ["model", "deformed", "vector", "contour", "distribution", "numeric_value_csv"] if enabled else [],
        "relative_displacement": bool(post.get("relative_displacement", enabled)),
        "legend": {
            "invert_supported": True,
            "default_high": "red",
            "default_low": "blue",
        },
        "report_manifest": enabled,
        "output_condition_manifest": enabled,
        "commercial_oss_roundtrip": False,
    }


def _mesh_public_diagnostics(mesh: Mesh2D) -> dict[str, Any]:
    element_types = sorted({element.type for element in mesh.elements})
    active_count = sum(1 for element in mesh.elements if element.active)
    return {
        "node_count": len(mesh.node_ids),
        "element_count": len(mesh.elements),
        "active_element_count": active_count,
        "element_types": element_types,
        "has_second_order": _mesh_has_second_order(mesh),
        "has_triangles": any(kind.startswith("TRI") for kind in element_types),
        "has_quads": any(kind.startswith("QUAD") for kind in element_types),
        "block_closure_diagnostics": "available",
        "gap_overlap_repair_candidates": "available",
    }


def _stress_release_metadata(stage_cfg: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    release = _normalized_release(stage_cfg.get("stress_release", stage_cfg.get("stress_release_ratio", stage_cfg.get("release_rate"))))
    if release is None:
        return {}
    workflow = str(stage_cfg.get("geofeas_workflow", stage_cfg.get("set", "global")) or "global")
    cumulative = 0.0
    for item in list(history) + [stage_cfg]:
        if not isinstance(item, Mapping):
            continue
        item_workflow = str(item.get("geofeas_workflow", item.get("set", "global")) or "global")
        if item_workflow != workflow:
            continue
        value = _normalized_release(item.get("stress_release", item.get("stress_release_ratio", item.get("release_rate"))))
        if value is not None:
            cumulative += value
    return {
        "stress_release": release,
        "cumulative_release": cumulative,
        "remaining_to_100_percent": max(0.0, 1.0 - cumulative),
        "release_ok": math.isclose(cumulative, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-9),
        "over_release": cumulative > 1.0 + 1.0e-9,
    }


def _normalized_release(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        release = float(value)
    except (TypeError, ValueError):
        return None
    if release > 1.0 and release <= 100.0:
        release /= 100.0
    return release


def _liquefaction_public_metadata(
    stage: StageResult2D,
    cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    workflow: str,
) -> dict[str, Any]:
    methods = _liquefaction_methods(materials)
    if workflow not in LIQUEFACTION_WORKFLOWS and not methods:
        return {}
    second_order = _mesh_has_second_order(mesh)
    hydro = _mapping(stage_cfg.get("hydro", stage_cfg.get("water", {})))
    has_pressure_load = bool(hydro.get("pressure_bcs") or hydro.get("pore_pressure_bcs") or hydro.get("seepage_csv") or stage.pore_pressure is not None)
    solver_liq = stage.solver_info.get("liquefaction", {})
    if not isinstance(solver_liq, Mapping):
        solver_liq = {}
    return {
        "required_stage_sequence": ["before_liquefaction", "during_liquefaction", "post_liquefaction_volume_compression"],
        "cyclic_stress_methods": methods or ["n_value_position"],
        "water_level_roles": {
            "fl_calculation": "separate_liquefaction_water_level",
            "water_pressure_load": "hydro_pressure_or_seepage_stage",
            "current_stage_has_pressure_load": has_pressure_load,
        },
        "second_order_required": True,
        "second_order_present": second_order,
        "second_order_warning": "" if second_order else "public guidance requires second-order elements for liquefaction checks",
        "min_FL": solver_liq.get("min_FL"),
        "max_ru": solver_liq.get("max_ru"),
        "post_outputs": ["liquefaction_state.csv", "liquefaction_history.csv", "liquefaction_ru_fl.svg", "liquefaction_post.html"],
    }


def _liquefaction_methods(materials: Mapping[str, ElasticPlaneStrainMaterial]) -> list[str]:
    methods: list[str] = []
    for material in materials.values():
        params = material.advanced_params or {}
        liq = params.get("liquefaction")
        liq_map = liq if isinstance(liq, Mapping) else params
        model = material.advanced_model or material.model
        if model in {"liquefaction", "bilinear_liquefaction"} or isinstance(liq, Mapping):
            method = str(_mapping(liq_map).get("cyclic_stress_method", "n_value_position") or "n_value_position")
            if method not in methods:
                methods.append(method)
    return methods


def _has_liquefaction_material(materials: Mapping[str, ElasticPlaneStrainMaterial]) -> bool:
    return bool(_liquefaction_methods(materials))


def _mesh_has_second_order(mesh: Mesh2D) -> bool:
    return bool(mesh.elements) and all(element.type in SECOND_ORDER_ELEMENTS for element in mesh.elements)


def _case_workflows(case: Mapping[str, Any]) -> list[str]:
    cfg = _mapping(case.get("config", {}))
    workflows: list[str] = []
    for stage in _stages_from_config(cfg):
        if isinstance(stage, Mapping) and stage.get("geofeas_workflow"):
            workflows.append(str(stage["geofeas_workflow"]))
    return workflows


def _stage_output_condition_files(stage: StageResult2D) -> list[str]:
    if stage.output_dir is None:
        return []
    candidates = [
        "report.html",
        "results.vtk",
        "displacements.csv",
        "element_stress.csv",
        "integration_point_stress.csv",
        "reactions.csv",
        "structural_section_forces.csv",
        "liquefaction_post.html",
        "liquefaction_history.csv",
    ]
    root = Path(stage.output_dir)
    return [name for name in candidates if (root / name).exists()]


def _matrix_row(
    row_id: str,
    priority: str,
    title: str,
    status: str,
    evidence: str,
    remaining_gap: str,
) -> dict[str, str]:
    return {
        "id": row_id,
        "priority": priority,
        "title": title,
        "public_status": status,
        "evidence": evidence,
        "remaining_gap": remaining_gap,
        "source_basis": "; ".join(source["url"] for source in PUBLIC_SOURCES),
    }


def _matrix_html(summary: Mapping[str, Any]) -> str:
    rows = summary.get("rows", [])
    table_rows = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("public_status", ""))
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('id', '')))}</td>"
            f"<td>{html.escape(str(row.get('priority', '')))}</td>"
            f"<td>{html.escape(str(row.get('title', '')))}</td>"
            f"<td class='{html.escape(status)}'>{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('evidence', '')))}</td>"
            f"<td>{html.escape(str(row.get('remaining_gap', '')))}</td>"
            "</tr>"
        )
    source_links = " ".join(f"<a href='{html.escape(src['url'])}'>{html.escape(src['label'])}</a>" for src in PUBLIC_SOURCES)
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='ja'>",
            "<head><meta charset='utf-8'><title>GeoFEAS Public Compatibility Matrix</title>",
            "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}th,td{border:1px solid #c7cfdd;padding:6px 8px;text-align:left;vertical-align:top}th{background:#eef2f8}.blocked_proprietary{color:#9a3412}.implemented_public,.covered_by_existing_regression{color:#166534}</style>",
            "</head><body>",
            "<h1>GeoFEAS 2D Public Compatibility Matrix</h1>",
            f"<p>Profile: {html.escape(str(summary.get('profile', PUBLIC_PROFILE)))} / rows: {html.escape(str(summary.get('row_count', 0)))}</p>",
            f"<p>Sources: {source_links}</p>",
            "<table><thead><tr><th>ID</th><th>Priority</th><th>Item</th><th>Status</th><th>Evidence</th><th>Remaining gap</th></tr></thead><tbody>",
            *table_rows,
            "</tbody></table>",
            "</body></html>",
        ]
    )


__all__ = [
    "PUBLIC_PROFILE",
    "PUBLIC_SOURCES",
    "WORKFLOW_CATALOG",
    "annotate_public_stage_result",
    "build_public_compatibility_matrix",
    "public_movie_tunnel_operation_steps",
    "public_output_conditions",
    "public_profile_enabled",
    "public_profile_run_warnings",
    "public_profile_summary",
    "public_workflow_operation_log",
    "workflow_catalog",
    "write_public_compatibility_matrix",
    "write_public_output_conditions",
    "write_public_profile_summary",
    "write_stage_public_profile",
]
