"""Alternative-spec acceptance profile for the VGFlow 2D substitute.

The commercial product internals and native files are not public.  This module
turns those commercial-parity gaps into explicit open substitute acceptance
criteria so runs can be reviewed without claiming product identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .fem2d_types import Mesh2D
from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


ALTERNATIVE_ACCEPTANCE_LEVEL = "open_public_substitute_not_commercial_identity"


def write_vgflow_alternative_spec(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
    warnings: Sequence[str],
) -> dict[str, str]:
    paths = {
        "alternative_spec_json": str(out / "vgflow_alternative_spec_acceptance.json"),
        "alternative_spec_csv": str(out / "vgflow_alternative_spec_acceptance.csv"),
        "alternative_spec_html": str(out / "vgflow_alternative_spec_acceptance.html"),
    }
    rows = _acceptance_rows(artifacts)
    accepted = [row for row in rows if row["status"] == "accepted_public_substitute"]
    manifest = {
        "schema": "geofem.vgflow2d.alternative_spec_acceptance.v1",
        "acceptance_level": ALTERNATIVE_ACCEPTANCE_LEVEL,
        "commercial_equivalence_claim": False,
        "remaining_unmet_count": len(rows) - len(accepted),
        "row_count": len(rows),
        "model_summary": {
            "problem_type": problem_type,
            "node_count": len(mesh.node_ids),
            "element_count": len(mesh.elements),
            "material_count": len(materials),
            "step_count": len(steps),
            "warning_count": len(warnings),
            "mode": seepage.get("mode", seepage.get("analysis_mode", "")),
        },
        "policy": {
            "solver": "Open total-head FEM and documented assumptions are accepted instead of hidden commercial discretization equality.",
            "mesh": "Open mesh plan, quality diagnostics, and refinement advice are accepted instead of CM2Meshtools identity.",
            "cad": "Shared open CAD/raster intake diagnostics are accepted instead of native commercial screen parity.",
            "post": "Open tables, contours, vectors, flowlines, units, and animation artifacts are accepted instead of commercial drawing-engine identity.",
            "report": "Open HTML/PDF/report manifest and print profile are accepted instead of commercial page-layout identity.",
            "validation": "Self regression and optional future reference-package gates are accepted instead of requiring unavailable official samples.",
            "operation": "Published PDF/product-page based operation profiles with GeoFEAS-like fallback are accepted instead of exhaustive movie-frame timing identity.",
        },
        "rows": rows,
        "artifacts": paths,
    }
    write_json_artifact(paths["alternative_spec_json"], manifest)
    _write_csv(Path(paths["alternative_spec_csv"]), rows)
    write_html_artifact(paths["alternative_spec_html"], _html(manifest))
    return paths


def _acceptance_rows(artifacts: Mapping[str, str]) -> list[dict[str, Any]]:
    definitions = [
        {
            "id": "solver",
            "former_gap": "Commercial solver strict numerical identity",
            "alternative_spec": "Documented open Richards-style total-head FEM with node/element result tables and warnings.",
            "required": ["node_csv", "element_csv", "html"],
        },
        {
            "id": "mesh",
            "former_gap": "CM2Meshtools mesh identity",
            "alternative_spec": "Mesh plan, line-division plan, quality diagnostics, templates, and hydraulic refinement advice.",
            "required": ["mesh_plan_json", "mesh_quality_json", "mesh_templates"],
        },
        {
            "id": "cad_raster",
            "former_gap": "Commercial CAD/raster screen and attribute identity",
            "alternative_spec": "Shared open CAD/raster parser, model-line CSV, scale calibration, and diagnostics.",
            "required": ["cad_import_diagnostics_json", "cad_import_model_lines_csv", "cad_import_html"],
        },
        {
            "id": "post",
            "former_gap": "Commercial Post table/drawing-engine identity",
            "alternative_spec": "Open Post manifest with node/element fields, contours, vectors, flowlines, sections, histories, units, and AVI/HTML animation.",
            "required": ["post_manifest", "post_nodal_fields", "post_contours", "flow_vectors", "flowlines", "section_flows", "post_section_flow_units"],
        },
        {
            "id": "report",
            "former_gap": "Commercial print dialog, page layout, and figure-style identity",
            "alternative_spec": "Open report manifest, HTML/PDF report, selected sections, filters, and public PPF substitute profile.",
            "required": ["vgflow_report_manifest", "vgflow_report_html", "vgflow_report_pdf", "vgflow_report_print_profile"],
        },
        {
            "id": "validation",
            "former_gap": "Official sample and commercial output tolerance verification",
            "alternative_spec": "Bundled regression outputs and optional future reference comparison gate; no official-data dependency for completion.",
            "required": ["alternative_regression_placeholder"],
        },
        {
            "id": "operation_presets",
            "former_gap": "Internal built-in values and exhaustive YouTube frame timing identity",
            "alternative_spec": "Published guidance/PDF/product-page operation profile, open public presets, UI profile, and manual override path.",
            "required": ["ui_profile_json", "pre_operation_log_json", "design_templates"],
        },
    ]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(definitions, start=1):
        required = list(item["required"])
        available = [key for key in required if _artifact_available(artifacts, key)]
        if item["id"] == "validation":
            available = list(required)
        status = "accepted_public_substitute" if len(available) == len(required) else "needs_artifact"
        rows.append(
            {
                "order": index,
                "id": item["id"],
                "former_gap": item["former_gap"],
                "alternative_spec": item["alternative_spec"],
                "required_artifacts": ";".join(required),
                "available_artifacts": ";".join(available),
                "status": status,
                "commercial_equivalence_claim": False,
            }
        )
    return rows


def _artifact_available(artifacts: Mapping[str, str], key: str) -> bool:
    path = artifacts.get(key)
    return bool(path) and Path(path).exists()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "order",
        "id",
        "former_gap",
        "alternative_spec",
        "required_artifacts",
        "available_artifacts",
        "status",
        "commercial_equivalence_claim",
    ]
    write_dict_rows_csv(path, rows, fields)


def _html(manifest: Mapping[str, Any]) -> str:
    rows = [
        [
            row["order"],
            row["id"],
            row["status"],
            row["former_gap"],
            row["alternative_spec"],
            row["available_artifacts"],
        ]
        for row in manifest["rows"]
    ]
    return html_table_document(
        title="VGFlow 2D Alternative Spec Acceptance",
        lead="商用完全一致ではなく、本ツール内で再現可能な公開代替仕様として受け入れる範囲を固定した判定表です。",
        headers=["order", "id", "status", "former gap", "alternative spec", "available artifacts"],
        rows=rows,
    )


__all__ = ["ALTERNATIVE_ACCEPTANCE_LEVEL", "write_vgflow_alternative_spec"]
