"""Reference-result comparison helpers for GeoFEAS style CSV outputs."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import geofeas_seepage as _geofeas_seepage


GEOFEAS_STRUCTURAL_COMPARISON_SPECS: dict[str, dict[str, Any]] = {
    "displacements.csv": {
        "key_fields": ("node_id",),
        "value_fields": ("ux", "uy", "u_norm", "settlement"),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "reactions.csv": {
        "key_fields": ("node_id", "dof"),
        "value_fields": ("reaction", "constrained_value"),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "structural_state.csv": {
        "key_fields": ("element_id",),
        "value_fields": (
            "axial_force",
            "shear_force",
            "shear_force_i",
            "shear_force_j",
            "spring_reaction",
            "end_moment_i",
            "end_moment_j",
            "rotation_i",
            "rotation_j",
            "axial_deformation",
            "shear_deformation",
        ),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "structural_section_forces.csv": {
        "key_fields": ("element_id", "x"),
        "value_fields": ("axial_force", "shear_force", "bending_moment"),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "interface_state.csv": {
        "key_fields": ("interface_id", "gp"),
        "value_fields": (
            "gap_t",
            "gap_n",
            "traction_t",
            "traction_n",
            "slip_abs",
            "opening",
            "closure",
            "friction_limit",
            "effective_roughness",
        ),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "load_combinations.csv": {
        "key_fields": ("combination", "case"),
        "value_fields": ("factor", "case_scale"),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "post_case_comparison.csv": {
        "key_fields": ("stage",),
        "value_fields": ("seismic_kh", "seismic_kv", "max_displacement", "max_settlement", "max_pore_pressure"),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
}

GEOFEAS_DYNAMIC_COMPARISON_SPECS: dict[str, dict[str, Any]] = {
    **GEOFEAS_STRUCTURAL_COMPARISON_SPECS,
    "dynamic_history.csv": {
        "key_fields": ("step",),
        "value_fields": (
            "time",
            "dt",
            "kh",
            "kv",
            "load_scale",
            "max_displacement",
            "max_velocity",
            "max_acceleration",
            "max_pore_pressure",
            "min_pore_pressure",
        ),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
    "pore_pressure.csv": {
        "key_fields": ("node_id",),
        "value_fields": ("pore_pressure", "time"),
        "rtol": 1.0e-5,
        "atol": 1.0e-7,
    },
}

EXTERNAL_SEEPAGE_PRODUCT_PROFILES: dict[str, dict[str, Any]] = {
    "generic_csv": {"delimiters": [",", "\t", ";"], "versions": ["generic"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
    "GeoFEAS": {"delimiters": [",", "\t"], "versions": ["auto"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
    "UC-1": {"delimiters": [",", "\t"], "versions": ["auto"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
    "VGFlow": {"delimiters": [",", "\t"], "versions": ["auto"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
}

GEOFEAS_PACKAGE_FILE_PROFILES: dict[str, dict[str, str]] = {
    ".csv": {"kind": "open_table", "public_status": "open_supported", "open_substitute": "CSV adapter", "recommended_action": "Use the CSV/reference comparison or seepage normalization tools."},
    ".tsv": {"kind": "open_table", "public_status": "open_supported", "open_substitute": "TSV/CSV adapter", "recommended_action": "Use the external seepage normalization tools when hydraulic fields are present."},
    ".txt": {"kind": "open_text", "public_status": "open_diagnostic", "open_substitute": "Text attachment", "recommended_action": "Archive with the public-profile manifest; parse manually if the schema is known."},
    ".dxf": {"kind": "cad", "public_status": "open_supported", "open_substitute": "DXF CAD adapter", "recommended_action": "Import through cad_import and verify layer/entity diagnostics."},
    ".sxf": {"kind": "cad", "public_status": "open_supported", "open_substitute": "SXF/P21 CAD adapter", "recommended_action": "Import through cad_import and verify attribute retention."},
    ".p21": {"kind": "cad", "public_status": "open_supported", "open_substitute": "SXF/P21 CAD adapter", "recommended_action": "Import through cad_import and verify attribute retention."},
    ".sfc": {"kind": "cad", "public_status": "open_supported", "open_substitute": "SXF/SFC CAD adapter", "recommended_action": "Import through cad_import and verify attribute retention."},
    ".gf1": {"kind": "geofeas_open_surrogate", "public_status": "open_diagnostic", "open_substitute": "GF1 payload diagnostics", "recommended_action": "Use the GF1 diagnostic reader and convert supported geometry/material payloads."},
    ".dwg": {"kind": "cad_binary", "public_status": "converter_required", "open_substitute": "DWG to DXF/SXF converter path", "recommended_action": "Convert with a licensed/open DWG converter, then import the resulting DXF/SXF and compare geometry counts."},
    ".gf2": {"kind": "geofeas_project", "public_status": "blocked_proprietary", "open_substitute": "YAML/JSON GeoFEM project plus public-profile manifest", "recommended_action": "Request exported CSV/DXF/SXF or recreate the model with the public workflow templates."},
    ".sta": {"kind": "geofeas_solver_input", "public_status": "blocked_proprietary", "open_substitute": "GeoFEM YAML input", "recommended_action": "Treat as reference attachment unless a stable public schema is supplied."},
    ".msh": {"kind": "geofeas_mesh_input", "public_status": "blocked_proprietary", "open_substitute": "GeoFEM mesh block YAML", "recommended_action": "Treat as reference attachment unless a stable public schema is supplied."},
    ".oss": {"kind": "geofeas_post_condition", "public_status": "blocked_proprietary", "open_substitute": "geofeas_public_output_conditions.json", "recommended_action": "Use the open Post output-condition JSON; native OSS roundtrip is not claimed."},
}

GEOFEAS_PUBLIC_MATERIAL_PROFILES: dict[str, dict[str, Any]] = {
    "elastic": {"category": "elastic", "public_status": "open_supported", "required_any": (("E", "young", "young_modulus"), ("nu", "poisson")), "recommended_action": "Use as ordinary linear elastic material."},
    "von_mises": {"category": "elasto_plastic", "public_status": "open_supported", "required_any": (("E",), ("nu",), ("yield_stress", "sigma_y")), "recommended_action": "Use as a public J2 substitute for strength-reduction and plastic benchmarks."},
    "j2": {"category": "elasto_plastic", "public_status": "open_supported", "required_any": (("E",), ("nu",), ("yield_stress", "sigma_y")), "recommended_action": "Use as a public J2 substitute for strength-reduction and plastic benchmarks."},
    "drucker_prager": {"category": "elasto_plastic", "public_status": "open_supported", "required_any": (("E",), ("nu",), ("cohesion", "c"), ("friction_angle", "phi")), "recommended_action": "Use as a public pressure-dependent substitute."},
    "dp": {"category": "elasto_plastic", "public_status": "open_supported", "required_any": (("E",), ("nu",), ("cohesion", "c"), ("friction_angle", "phi")), "recommended_action": "Use as a public pressure-dependent substitute."},
    "mohr_coulomb": {"category": "elasto_plastic", "public_status": "open_supported", "required_any": (("E",), ("nu",), ("cohesion", "c"), ("friction_angle", "phi")), "recommended_action": "Use as a public MC-like substitute; exact GeoFEAS active-set details need reference comparison."},
    "mc": {"category": "elasto_plastic", "public_status": "open_supported", "required_any": (("E",), ("nu",), ("cohesion", "c"), ("friction_angle", "phi")), "recommended_action": "Use as a public MC-like substitute; exact GeoFEAS active-set details need reference comparison."},
    "nonlinear_elastic": {"category": "advanced", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50")), "recommended_action": "Check stiffness-reduction curve against lab data or public benchmark."},
    "hardin_drnevich": {"category": "advanced", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50")), "recommended_action": "Use as public Hardin-Drnevich style substitute; verify G/G0 curve."},
    "ramberg_osgood": {"category": "advanced", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50"), ("alpha", "a"), ("r", "n", "exponent")), "recommended_action": "Use as public Ramberg-Osgood style substitute; verify curve fit report."},
    "uw_clay": {"category": "advanced_clay", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50"), ("su", "cu", "undrained_shear_strength")), "recommended_action": "Treat as DP/J2-mapped public substitute unless a GeoFEAS/UWLC reference model is supplied."},
    "pastor_zienkiewicz_sand": {"category": "advanced_sand", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50"), ("friction_angle", "phi_cs", "critical_state_phi")), "recommended_action": "Treat as public PZ-style substitute; verify dilatancy and cyclic history against reference data."},
    "pastor_zienkiewicz_clay": {"category": "advanced_clay", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50"), ("su", "cu", "undrained_shear_strength")), "recommended_action": "Treat as public PZ-style substitute; verify hardening and pore-pressure history against reference data."},
    "liquefaction": {"category": "liquefaction", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50"), ("cyclic_resistance_ratio", "CRR", "RL", "RL20"), ("cyclic_stress_ratio", "CSR", "L")), "recommended_action": "Use H19/H28 public workflow metadata and compare FL/ru histories when reference data exists."},
    "bilinear_liquefaction": {"category": "liquefaction", "public_status": "public_substitute", "required_any": (("G0", "Gmax", "initial_shear_modulus"), ("gamma_ref", "reference_strain", "gamma50"), ("cyclic_resistance_ratio", "CRR", "RL", "RL20"), ("cyclic_stress_ratio", "CSR", "L")), "recommended_action": "Use H19/H28 public workflow metadata and compare FL/ru histories when reference data exists."},
}

EXTERNAL_PRODUCT_VERSION_PROFILES: dict[str, dict[str, Any]] = {
    "GeoFEAS": {
        "aliases": ("geofeas", "geo_feas", "geo-feas", "geo2d", "gf2"),
        "public_versions": ("auto", "v5_public"),
        "open_extensions": (".csv", ".tsv", ".dxf", ".sxf", ".p21", ".sfc", ".gf1"),
        "private_extensions": (".gf2", ".sta", ".msh", ".oss"),
        "recommended_action": "Use public CSV/DXF/SXF/P21/GF1 substitutes; keep private GeoFEAS files as reference attachments.",
    },
    "UC-1": {
        "aliases": ("uc-1", "uc1", "u c 1", "uc_1"),
        "public_versions": ("auto",),
        "open_extensions": (".csv", ".tsv", ".txt"),
        "private_extensions": (),
        "recommended_action": "Use open CSV/TSV exports and compare normalized hydraulic/result columns.",
    },
    "VGFlow": {
        "aliases": ("vgflow", "vg_flow", "vg-flow"),
        "public_versions": ("auto",),
        "open_extensions": (".csv", ".tsv", ".txt"),
        "private_extensions": (),
        "recommended_action": "Use open seepage-result exports; validate head/water-pressure unit conversion.",
    },
    "UWLC": {
        "aliases": ("uwlc", "uw_clay", "uw-clay", "uw clay"),
        "public_versions": ("auto",),
        "open_extensions": (".csv", ".tsv", ".txt"),
        "private_extensions": (),
        "recommended_action": "Use open material/result tables and material-profile audits; exact UWLC internals require reference data.",
    },
    "generic_csv": {
        "aliases": ("csv", "tsv"),
        "public_versions": ("generic", "auto"),
        "open_extensions": (".csv", ".tsv"),
        "private_extensions": (),
        "recommended_action": "Treat as an open interchange file and map recognized headers.",
    },
}


_SEEPAGE_HEADER_ALIASES = {
    "time": {"time", "t", "step_time", "elapsed_time", "時刻", "時間", "経過時間"},
    "step": {"step", "stage", "ステップ", "段階"},
    "node_id": {"node_id", "node", "node_no", "node_number", "節点", "節点番号", "節点no", "節点No", "No"},
    "x": {"x", "x座標", "座標x"},
    "y": {"y", "y座標", "座標y"},
    "pore_pressure": {"pore_pressure", "pressure", "p", "water_pressure", "水圧", "間隙水圧", "過剰間隙水圧"},
    "head": {"head", "water_head", "total_head", "全水頭", "水頭", "水位標高"},
    "water_level": {"water_level", "level", "水位"},
    "unit": {"unit", "単位"},
}


def import_external_seepage_results(path: str | Path, *, product: str = "auto") -> list[dict[str, Any]]:
    """Read GeoFEAS/UC-1 style seepage CSV/TSV output into normalized rows."""

    return _geofeas_seepage.import_external_seepage_results(path, product=product)


def compare_external_seepage_results(
    actual_path: str | Path,
    reference_path: str | Path,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Compare normalized external seepage results by time and node."""

    return _geofeas_seepage.compare_external_seepage_results(actual_path, reference_path, rtol=rtol, atol=atol)


def compare_geofeas_reference_csv(
    actual_csv: str | Path,
    reference_csv: str | Path,
    *,
    key_fields: Iterable[str] | None = None,
    value_fields: Iterable[str] | None = None,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Compare an output CSV with a GeoFEAS/reference CSV using keyed numeric fields."""

    actual_rows = _read_csv(actual_csv)
    reference_rows = _read_csv(reference_csv)
    keys = list(key_fields or _guess_key_fields(actual_rows, reference_rows))
    if not keys:
        raise ValueError("key_fields could not be inferred")
    fields = list(value_fields or _guess_value_fields(actual_rows, reference_rows, keys))
    actual = {_row_key(row, keys): row for row in actual_rows}
    reference = {_row_key(row, keys): row for row in reference_rows}
    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))
    comparisons: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    failed = 0
    for key in sorted(set(actual) & set(reference)):
        for field in fields:
            if field not in actual[key] or field not in reference[key]:
                continue
            try:
                av = float(actual[key][field])
                rv = float(reference[key][field])
            except (TypeError, ValueError):
                continue
            abs_err = abs(av - rv)
            rel_err = abs_err / max(abs(rv), atol)
            ok = abs_err <= atol + rtol * abs(rv)
            failed += 0 if ok else 1
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            comparisons.append({"key": key, "field": field, "actual": av, "reference": rv, "abs_error": abs_err, "rel_error": rel_err, "ok": ok})
    return {
        "actual": str(actual_csv),
        "reference": str(reference_csv),
        "key_fields": keys,
        "value_fields": fields,
        "rtol": rtol,
        "atol": atol,
        "row_count": len(comparisons),
        "missing_keys": missing,
        "extra_keys": extra,
        "failed_count": failed + len(missing),
        "passed": failed == 0 and not missing,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "comparisons": comparisons,
    }


def compare_geofeas_stage_package(
    actual_dir: str | Path,
    reference_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare a whole GeoFEM stage output folder with a GeoFEAS/reference folder."""

    actual_root = Path(actual_dir)
    reference_root = Path(reference_dir)
    out_root = Path(output_dir) if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
    comparison_specs = specs or GEOFEAS_STRUCTURAL_COMPARISON_SPECS
    files: list[dict[str, Any]] = []
    failed_count = 0
    compared_count = 0
    max_abs = 0.0
    max_rel = 0.0
    for filename, spec in comparison_specs.items():
        actual = actual_root / filename
        reference = reference_root / filename
        if not reference.exists():
            files.append({"file": filename, "status": "reference_missing", "passed": True, "row_count": 0, "failed_count": 0})
            continue
        if not actual.exists():
            files.append({"file": filename, "status": "actual_missing", "passed": False, "row_count": 0, "failed_count": 1})
            failed_count += 1
            continue
        summary = compare_geofeas_reference_csv(
            actual,
            reference,
            key_fields=spec.get("key_fields"),
            value_fields=spec.get("value_fields"),
            rtol=float(spec.get("rtol", 1.0e-5)),
            atol=float(spec.get("atol", 1.0e-7)),
        )
        summary["file"] = filename
        summary["status"] = "compared"
        files.append(summary)
        compared_count += 1
        failed_count += int(summary.get("failed_count", 0) or 0)
        max_abs = max(max_abs, float(summary.get("max_abs_error", 0.0) or 0.0))
        max_rel = max(max_rel, float(summary.get("max_rel_error", 0.0) or 0.0))
        if out_root is not None:
            stem = Path(filename).stem
            write_geofeas_comparison_csv(summary, out_root / f"{stem}_comparison.csv")
            write_geofeas_tolerance_report(summary, out_root / f"{stem}_tolerance.html")
    package = {
        "actual_dir": str(actual_root),
        "reference_dir": str(reference_root),
        "compared_count": compared_count,
        "failed_count": failed_count,
        "passed": failed_count == 0,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "files": files,
    }
    if out_root is not None:
        write_geofeas_package_report(package, out_root / "geofeas_package_tolerance.html")
    return package


def compare_geofeas_dynamic_sample(
    actual_dir: str | Path,
    reference_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare dynamic-stage outputs against a GeoFEAS/reference dynamic sample package."""

    return compare_geofeas_stage_package(actual_dir, reference_dir, output_dir=output_dir, specs=specs or GEOFEAS_DYNAMIC_COMPARISON_SPECS)


def write_external_seepage_results(rows: Iterable[Mapping[str, Any]], path: str | Path, *, product: str = "generic_csv") -> None:
    """Write normalized seepage rows in the open interchange CSV used for round-trip checks."""

    _geofeas_seepage.write_external_seepage_results(rows, path, product=product)


def compare_external_seepage_roundtrip(
    source_path: str | Path,
    exported_path: str | Path,
    *,
    product: str = "auto",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Normalize, export, and compare an external seepage file through the open interchange path."""

    return _geofeas_seepage.compare_external_seepage_roundtrip(source_path, exported_path, product=product, rtol=rtol, atol=atol)


def diagnose_external_seepage_version(path: str | Path, *, product: str = "auto") -> dict[str, Any]:
    """Report detected product/version traits and unmapped columns for a seepage result file."""

    return _geofeas_seepage.diagnose_external_seepage_version(path, product=product)


def diagnose_geofeas_package_files(package_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Inventory a GeoFEAS/GeoFEM exchange folder and classify public compatibility."""

    root = Path(package_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        profile = GEOFEAS_PACKAGE_FILE_PROFILES.get(suffix, {})
        status = profile.get("public_status", "unknown")
        size = path.stat().st_size
        row = {
            "relative_path": path.relative_to(root).as_posix(),
            "extension": suffix or "(none)",
            "size_bytes": size,
            "kind": profile.get("kind", "unknown"),
            "public_status": status,
            "open_substitute": profile.get("open_substitute", ""),
            "recommended_action": profile.get("recommended_action", "Inspect manually and add a format profile before claiming compatibility."),
            "binary_hint": _binary_hint(path),
            "blocked_reason": _package_blocked_reason(suffix, status),
        }
        rows.append(row)
    status_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["public_status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    summary = {
        "schema": "geofem.geofeas_package_inventory.v1",
        "package_dir": str(root),
        "file_count": len(rows),
        "status_counts": status_counts,
        "blocked_count": status_counts.get("blocked_proprietary", 0),
        "converter_required_count": status_counts.get("converter_required", 0),
        "open_supported_count": status_counts.get("open_supported", 0),
        "unknown_count": status_counts.get("unknown", 0),
        "roundtrip_claim": "open_substitute_only",
        "native_private_roundtrip": False,
        "rows": rows,
    }
    if output_dir is not None:
        write_geofeas_package_inventory(summary, output_dir)
    return summary


def verify_public_workflow_log(
    workflow: str,
    actual_log_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    min_action_token_overlap: float = 0.35,
) -> dict[str, Any]:
    """Compare an observed GUI operation log with the public GeoFEAS workflow template."""

    from .geofeas_public import public_workflow_operation_log

    actual_path = Path(actual_log_path)
    expected = [_normalize_operation_row(row, index) for index, row in enumerate(public_workflow_operation_log(workflow), start=1)]
    actual = [_normalize_operation_row(row, index) for index, row in enumerate(_read_operation_log(actual_path), start=1)]
    comparisons: list[dict[str, Any]] = []
    cursor = 0
    used_actual: set[int] = set()
    for expected_row in expected:
        best_index = -1
        best_score = -1.0
        best_tab_match = False
        for index in range(cursor, len(actual)):
            actual_row = actual[index]
            overlap = _operation_token_overlap(expected_row["action"], actual_row["action"])
            tab_match = _casefold(expected_row["tab"]) == _casefold(actual_row["tab"])
            score = overlap + (0.25 if tab_match else 0.0)
            if score > best_score:
                best_index = index
                best_score = score
                best_tab_match = tab_match
        if best_index >= 0:
            actual_row = actual[best_index]
            overlap = _operation_token_overlap(expected_row["action"], actual_row["action"])
            passed = best_tab_match and overlap >= min_action_token_overlap
            status = "matched" if passed else ("tab_mismatch" if overlap >= min_action_token_overlap else "action_mismatch")
            if passed:
                cursor = best_index + 1
                used_actual.add(best_index)
            comparisons.append(
                {
                    "expected_step": expected_row["step"],
                    "status": status,
                    "passed": passed,
                    "expected_tab": expected_row["tab"],
                    "actual_tab": actual_row["tab"],
                    "expected_action": expected_row["action"],
                    "actual_action": actual_row["action"],
                    "action_token_overlap": overlap,
                    "actual_index": best_index + 1,
                }
            )
        else:
            comparisons.append(
                {
                    "expected_step": expected_row["step"],
                    "status": "missing",
                    "passed": False,
                    "expected_tab": expected_row["tab"],
                    "actual_tab": "",
                    "expected_action": expected_row["action"],
                    "actual_action": "",
                    "action_token_overlap": 0.0,
                    "actual_index": None,
                }
            )
    extra_rows = [actual[index] for index in range(len(actual)) if index not in used_actual]
    failed = [row for row in comparisons if not bool(row.get("passed"))]
    summary = {
        "schema": "geofem.geofeas_public_workflow_log_verification.v1",
        "workflow": workflow,
        "actual_log": str(actual_path),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "matched_count": sum(1 for row in comparisons if row["status"] == "matched"),
        "failed_count": len(failed),
        "extra_count": len(extra_rows),
        "min_action_token_overlap": min_action_token_overlap,
        "passed": not failed,
        "comparisons": comparisons,
        "extra_operations": extra_rows,
        "remaining_gap": "This verifies public workflow order and operation wording only; commercial pixel/UI-state parity still requires licensed captures.",
    }
    if output_dir is not None:
        write_public_workflow_log_verification(summary, output_dir)
    return summary


def write_public_workflow_log_verification(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write workflow-log verification JSON/CSV/HTML artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "geofeas_public_workflow_log_verification.json"
    csv_path = out / "geofeas_public_workflow_log_verification.csv"
    html_path = out / "geofeas_public_workflow_log_verification.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fieldnames = [
        "expected_step",
        "status",
        "passed",
        "expected_tab",
        "actual_tab",
        "action_token_overlap",
        "expected_action",
        "actual_action",
        "actual_index",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("comparisons", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    html_path.write_text(_workflow_log_verification_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def audit_public_material_profile(config: Mapping[str, Any], *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Audit material definitions against the public GeoFEAS substitute profile."""

    materials = config.get("materials", config.get("material", {}))
    if not isinstance(materials, Mapping):
        materials = {}
    rows: list[dict[str, Any]] = []
    for name, raw in sorted(materials.items(), key=lambda item: str(item[0])):
        spec = raw if isinstance(raw, Mapping) else {}
        model = str(spec.get("model", "elastic") or "elastic").strip().lower()
        profile = GEOFEAS_PUBLIC_MATERIAL_PROFILES.get(model)
        flattened = _flatten_material_spec(spec)
        if profile is None:
            row = {
                "material": str(name),
                "model": model,
                "category": "unknown",
                "public_status": "unsupported_or_unknown",
                "severity": "ERROR",
                "missing_required": "",
                "present_public_fields": ",".join(sorted(flattened)),
                "geofeas_private_gap": "No public GeoFEM material profile is registered for this model.",
                "recommended_action": "Map to a supported public substitute or add a verified material profile before claiming GeoFEAS equivalence.",
            }
            rows.append(row)
            continue
        missing_groups = [_alias_group_label(group) for group in profile.get("required_any", ()) if not _has_any_alias(flattened, group)]
        status = str(profile.get("public_status", "public_substitute"))
        private_gap = _material_private_gap(model, status)
        severity = "ERROR" if missing_groups else ("WARN" if status == "public_substitute" else "INFO")
        row = {
            "material": str(name),
            "model": model,
            "category": str(profile.get("category", "")),
            "public_status": status,
            "severity": severity,
            "missing_required": "; ".join(missing_groups),
            "present_public_fields": ",".join(sorted(flattened)),
            "geofeas_private_gap": private_gap,
            "recommended_action": str(profile.get("recommended_action", "")),
        }
        rows.append(row)
    stage_workflows = _material_stage_workflows(config)
    liquefaction_rows = [row for row in rows if row["category"] == "liquefaction"]
    warnings: list[str] = []
    if liquefaction_rows and not any(workflow in {"river_liquefaction_h19", "river_liquefaction_h28"} for workflow in stage_workflows):
        warnings.append("Liquefaction materials are present, but no H19/H28 river liquefaction workflow stage was detected.")
    if any(row["severity"] == "ERROR" for row in rows):
        warnings.append("One or more materials are missing required public-profile parameters or use unknown models.")
    summary = {
        "schema": "geofem.geofeas_public_material_audit.v1",
        "material_count": len(rows),
        "error_count": sum(1 for row in rows if row["severity"] == "ERROR"),
        "warning_count": sum(1 for row in rows if row["severity"] == "WARN"),
        "public_substitute_count": sum(1 for row in rows if row["public_status"] == "public_substitute"),
        "open_supported_count": sum(1 for row in rows if row["public_status"] == "open_supported"),
        "stage_workflows": stage_workflows,
        "passed": not any(row["severity"] == "ERROR" for row in rows),
        "native_geo_feas_material_equivalence": False,
        "remaining_gap": "Exact GeoFEAS internal formulas, hidden defaults, hardening/dilatancy laws, cyclic history rules, and product-version-specific behavior remain unverified without official references.",
        "warnings": warnings,
        "rows": rows,
    }
    if output_dir is not None:
        write_public_material_audit(summary, output_dir)
    return summary


def write_public_material_audit(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write material-profile audit JSON/CSV/HTML artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "geofeas_public_material_audit.json"
    csv_path = out / "geofeas_public_material_audit.csv"
    html_path = out / "geofeas_public_material_audit.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fieldnames = [
        "material",
        "model",
        "category",
        "public_status",
        "severity",
        "missing_required",
        "present_public_fields",
        "geofeas_private_gap",
        "recommended_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("rows", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    html_path.write_text(_material_audit_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def audit_post_report_package(
    result_dir: str | Path,
    *,
    baseline_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit Post/report artifacts against the public GeoFEAS output profile."""

    root = Path(result_dir)
    baseline = Path(baseline_dir) if baseline_dir is not None else None
    checks: list[dict[str, Any]] = []

    _add_file_check(checks, root / "calculation_report.html", "report.html", "ERROR", "HTML calculation report")
    _add_file_check(checks, root / "calculation_report.pdf", "report.pdf", "ERROR", "Direct PDF calculation report")
    _add_file_check(checks, root / "calculation_report_manifest.json", "report.manifest", "ERROR", "Reproducibility/report manifest")
    _add_file_check(checks, root / "calculation_report_input_snapshot.json", "report.input_snapshot", "WARN", "Frozen input snapshot")
    _add_file_check(checks, root / "geofeas_public_output_conditions.json", "post.output_conditions", "WARN", "Open Post output-condition substitute")

    svg_files = sorted(path for path in root.rglob("*.svg") if path.is_file())
    png_files = sorted(path for path in root.rglob("*.png") if path.is_file())
    html_post_files = sorted(path for path in root.rglob("*post*.html") if path.is_file())
    csv_files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    value_csv = [path for path in csv_files if path.name in _public_post_value_csv_names() or "value" in path.stem.lower()]
    _add_count_check(checks, "post.svg_figures", len(svg_files), 1, "ERROR", "At least one vector Post figure")
    _add_count_check(checks, "post.value_csv", len(value_csv), 1, "WARN", "Value-check/result CSV outputs")
    _add_count_check(checks, "post.html_views", len(html_post_files), 0, "INFO", "HTML Post detail views")
    _add_count_check(checks, "post.png_exports", len(png_files), 0, "INFO", "PNG exports")

    manifest = _read_json_if_exists(root / "calculation_report_manifest.json")
    if manifest:
        features = manifest.get("features", [])
        feature_text = ",".join(str(item) for item in features) if isinstance(features, list) else str(features)
        checks.append(
            {
                "id": "report.manifest_features",
                "severity": "INFO",
                "passed": True,
                "actual": feature_text,
                "expected": "manifest readable",
                "detail": "Report manifest was parsed.",
            }
        )
    output_conditions = _read_json_if_exists(root / "geofeas_public_output_conditions.json")
    if output_conditions:
        save = output_conditions.get("save_behavior", {}) if isinstance(output_conditions.get("save_behavior", {}), Mapping) else {}
        checks.append(
            {
                "id": "post.output_conditions_native_oss",
                "severity": "WARN",
                "passed": not bool(save.get("commercial_oss_roundtrip", False)),
                "actual": bool(save.get("commercial_oss_roundtrip", False)),
                "expected": False,
                "detail": "Open JSON output-condition substitute is present; native OSS parity is not claimed.",
            }
        )

    current_visuals = [*svg_files, *png_files]
    if baseline is None:
        checks.append(
            {
                "id": "post.visual_baseline",
                "severity": "WARN",
                "passed": False,
                "actual": "not provided",
                "expected": "commercial or accepted visual baseline directory",
                "detail": "Pixel/visual parity cannot be asserted without baselines and tolerances.",
            }
        )
    else:
        missing = [path.relative_to(root).as_posix() for path in current_visuals if not (baseline / path.relative_to(root)).exists()]
        checks.append(
            {
                "id": "post.visual_baseline_files",
                "severity": "WARN" if missing else "INFO",
                "passed": not missing,
                "actual": len(current_visuals) - len(missing),
                "expected": len(current_visuals),
                "detail": "Missing visual baselines: " + ", ".join(missing[:20]) if missing else "All current visual files have matching baseline paths.",
            }
        )

    errors = [row for row in checks if row["severity"] == "ERROR" and not bool(row["passed"])]
    warnings = [row for row in checks if row["severity"] == "WARN" and not bool(row["passed"])]
    summary = {
        "schema": "geofem.geofeas_post_report_audit.v1",
        "result_dir": str(root),
        "baseline_dir": str(baseline) if baseline is not None else "",
        "svg_count": len(svg_files),
        "png_count": len(png_files),
        "csv_count": len(csv_files),
        "value_csv_count": len(value_csv),
        "post_html_count": len(html_post_files),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "passed": not errors,
        "pixel_equivalent_post_claim": False,
        "native_oss_roundtrip_claim": False,
        "remaining_gap": "Pixel-equivalent Post views, product-version legend defaults, official report templates, and native OSS compatibility still require commercial examples and accepted visual tolerances.",
        "visual_files": [path.relative_to(root).as_posix() for path in current_visuals],
        "value_csv_files": [path.relative_to(root).as_posix() for path in value_csv],
        "checks": checks,
    }
    if output_dir is not None:
        write_post_report_audit(summary, output_dir)
    return summary


def audit_external_product_versions(package_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Audit open/proprietary external product files and version/header parity limits."""

    root = Path(package_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        package_profile = GEOFEAS_PACKAGE_FILE_PROFILES.get(ext, {})
        raw_rows: list[dict[str, str]] = []
        headers: list[str] = []
        mapped_headers: dict[str, str] = {}
        unsupported_headers: list[str] = []
        delimited_read_error = ""
        if ext in {".csv", ".tsv", ".txt"} and _binary_hint(path) == "text":
            try:
                raw_rows = _read_delimited_rows(path)
                headers = list(raw_rows[0].keys() if raw_rows else [])
                mapped_headers = {header: _normalize_external_header(header) for header in headers}
                unsupported_headers = [header for header, normalized in mapped_headers.items() if not normalized]
            except Exception as exc:  # pragma: no cover - defensive for real-world mixed encodings
                delimited_read_error = str(exc)
        product = _guess_external_product(path, headers)
        product_profile = EXTERNAL_PRODUCT_VERSION_PROFILES.get(product, EXTERNAL_PRODUCT_VERSION_PROFILES["generic_csv"])
        version = _detect_external_version(path, raw_rows)
        version_status = _external_version_status(version, product_profile)
        public_status = _external_file_public_status(ext, package_profile, product_profile)
        field_status = _external_field_mapping_status(headers, mapped_headers, unsupported_headers, delimited_read_error)
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "extension": ext or "(none)",
                "detected_product": product,
                "detected_version": version,
                "version_status": version_status,
                "public_status": public_status,
                "field_mapping_status": field_status,
                "headers": ",".join(headers),
                "mapped_headers": json.dumps(mapped_headers, ensure_ascii=False, sort_keys=True),
                "unsupported_headers": ",".join(unsupported_headers),
                "binary_hint": _binary_hint(path),
                "open_substitute": package_profile.get("open_substitute", "open CSV/TSV diagnostic path"),
                "recommended_action": product_profile.get("recommended_action", package_profile.get("recommended_action", "")),
                "blocked_reason": _external_product_blocked_reason(ext, public_status, version_status),
            }
        )
    status_counts: dict[str, int] = {}
    product_counts: dict[str, int] = {}
    version_warnings = 0
    for row in rows:
        status = str(row["public_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        product = str(row["detected_product"])
        product_counts[product] = product_counts.get(product, 0) + 1
        if row["version_status"] != "public_profile_available":
            version_warnings += 1
    summary = {
        "schema": "geofem.external_product_version_audit.v1",
        "package_dir": str(root),
        "file_count": len(rows),
        "product_counts": product_counts,
        "status_counts": status_counts,
        "version_warning_count": version_warnings,
        "blocked_count": status_counts.get("blocked_proprietary", 0),
        "converter_required_count": status_counts.get("converter_required", 0),
        "open_supported_count": status_counts.get("open_supported", 0),
        "unknown_count": status_counts.get("unknown", 0),
        "exact_product_version_parity": False,
        "remaining_gap": "Exact VGFlow/GeoFEAS/UWLC/UC-1 native exchange behavior, DWG converter differences, and version-specific attribute preservation remain dependent on unavailable product data.",
        "rows": rows,
    }
    if output_dir is not None:
        write_external_product_version_audit(summary, output_dir)
    return summary


def write_external_product_version_audit(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write external-product version audit JSON/CSV/HTML artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "external_product_version_audit.json"
    csv_path = out / "external_product_version_audit.csv"
    html_path = out / "external_product_version_audit.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fieldnames = [
        "relative_path",
        "extension",
        "detected_product",
        "detected_version",
        "version_status",
        "public_status",
        "field_mapping_status",
        "headers",
        "unsupported_headers",
        "binary_hint",
        "open_substitute",
        "blocked_reason",
        "recommended_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("rows", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    html_path.write_text(_external_product_version_audit_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def write_post_report_audit(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write Post/report audit JSON/CSV/HTML artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "geofeas_post_report_audit.json"
    csv_path = out / "geofeas_post_report_audit.csv"
    html_path = out / "geofeas_post_report_audit.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fieldnames = ["id", "severity", "passed", "actual", "expected", "detail"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("checks", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    html_path.write_text(_post_report_audit_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def write_geofeas_package_inventory(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write package-inventory JSON/CSV/HTML artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "geofeas_package_inventory.json"
    csv_path = out / "geofeas_package_inventory.csv"
    html_path = out / "geofeas_package_inventory.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fieldnames = ["relative_path", "extension", "size_bytes", "kind", "public_status", "binary_hint", "open_substitute", "blocked_reason", "recommended_action"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("rows", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    html_path.write_text(_package_inventory_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def write_geofeas_comparison_csv(summary: Mapping[str, Any], path: str | Path) -> None:
    rows = [row for row in summary.get("comparisons", []) if isinstance(row, Mapping)]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["key", "field", "actual", "reference", "abs_error", "rel_error", "ok"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_geofeas_tolerance_report(summary: Mapping[str, Any], path: str | Path) -> None:
    rows = [row for row in summary.get("comparisons", []) if isinstance(row, Mapping)]
    status = "PASSED" if bool(summary.get("passed", False)) else "FAILED"
    body_rows = []
    for row in rows:
        ok = bool(row.get("ok", False))
        body_rows.append(
            "<tr class='{cls}'><td>{key}</td><td>{field}</td><td>{actual:.12g}</td><td>{reference:.12g}</td>"
            "<td>{abs_error:.6g}</td><td>{rel_error:.6g}</td><td>{ok}</td></tr>".format(
                cls="ok" if ok else "ng",
                key=html.escape(str(row.get("key", ""))),
                field=html.escape(str(row.get("field", ""))),
                actual=float(row.get("actual", 0.0) or 0.0),
                reference=float(row.get("reference", 0.0) or 0.0),
                abs_error=float(row.get("abs_error", 0.0) or 0.0),
                rel_error=float(row.get("rel_error", 0.0) or 0.0),
                ok="OK" if ok else "NG",
            )
        )
    missing = "".join(f"<li>{html.escape(str(key))}</li>" for key in summary.get("missing_keys", []))
    extra = "".join(f"<li>{html.escape(str(key))}</li>" for key in summary.get("extra_keys", []))
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS tolerance comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.ng {{ background: #fef2f2; }}
.status {{ font-weight: 700; }}
</style>
</head>
<body>
<h1>GeoFEAS tolerance comparison</h1>
<p class="status">Status: {status}</p>
<p>actual: {html.escape(str(summary.get("actual", "")))}<br>reference: {html.escape(str(summary.get("reference", "")))}</p>
<p>rtol={float(summary.get("rtol", 0.0) or 0.0):.6g}, atol={float(summary.get("atol", 0.0) or 0.0):.6g},
failed_count={int(summary.get("failed_count", 0) or 0)}, max_abs_error={float(summary.get("max_abs_error", 0.0) or 0.0):.6g},
max_rel_error={float(summary.get("max_rel_error", 0.0) or 0.0):.6g}</p>
<h2>Comparison rows</h2>
<table>
<thead><tr><th>key</th><th>field</th><th>actual</th><th>reference</th><th>abs_error</th><th>rel_error</th><th>judgement</th></tr></thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
<h2>Missing keys</h2>
<ul>{missing}</ul>
<h2>Extra keys</h2>
<ul>{extra}</ul>
</body>
</html>
"""
    Path(path).write_text(document, encoding="utf-8")


def write_geofeas_package_report(summary: Mapping[str, Any], path: str | Path) -> None:
    status = "PASSED" if bool(summary.get("passed", False)) else "FAILED"
    rows: list[str] = []
    for file_summary in summary.get("files", []):
        if not isinstance(file_summary, Mapping):
            continue
        ok = bool(file_summary.get("passed", False))
        rows.append(
            "<tr class='{cls}'><td>{file}</td><td>{status}</td><td>{rows}</td><td>{failed}</td>"
            "<td>{max_abs:.6g}</td><td>{max_rel:.6g}</td></tr>".format(
                cls="ok" if ok else "ng",
                file=html.escape(str(file_summary.get("file", ""))),
                status=html.escape(str(file_summary.get("status", ""))),
                rows=int(file_summary.get("row_count", 0) or 0),
                failed=int(file_summary.get("failed_count", 0) or 0),
                max_abs=float(file_summary.get("max_abs_error", 0.0) or 0.0),
                max_rel=float(file_summary.get("max_rel_error", 0.0) or 0.0),
            )
        )
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS package tolerance comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.ng {{ background: #fef2f2; }}
.status {{ font-weight: 700; }}
</style>
</head>
<body>
<h1>GeoFEAS package tolerance comparison</h1>
<p class="status">Status: {status}</p>
<p>actual_dir: {html.escape(str(summary.get("actual_dir", "")))}<br>reference_dir: {html.escape(str(summary.get("reference_dir", "")))}</p>
<p>compared_count={int(summary.get("compared_count", 0) or 0)}, failed_count={int(summary.get("failed_count", 0) or 0)},
max_abs_error={float(summary.get("max_abs_error", 0.0) or 0.0):.6g},
max_rel_error={float(summary.get("max_rel_error", 0.0) or 0.0):.6g}</p>
<table>
<thead><tr><th>file</th><th>status</th><th>rows</th><th>failed</th><th>max_abs_error</th><th>max_rel_error</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""
    Path(path).write_text(document, encoding="utf-8")


def _package_inventory_html(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for row in summary.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("public_status", "unknown"))
        cls = {
            "open_supported": "ok",
            "open_diagnostic": "diag",
            "converter_required": "warn",
            "blocked_proprietary": "ng",
        }.get(status, "unknown")
        rows.append(
            "<tr class='{cls}'><td>{path}</td><td>{ext}</td><td>{kind}</td><td>{status}</td>"
            "<td>{sub}</td><td>{reason}</td><td>{action}</td></tr>".format(
                cls=cls,
                path=html.escape(str(row.get("relative_path", ""))),
                ext=html.escape(str(row.get("extension", ""))),
                kind=html.escape(str(row.get("kind", ""))),
                status=html.escape(status),
                sub=html.escape(str(row.get("open_substitute", ""))),
                reason=html.escape(str(row.get("blocked_reason", ""))),
                action=html.escape(str(row.get("recommended_action", ""))),
            )
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS package inventory</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.diag {{ background: #eff6ff; }}
.warn {{ background: #fffbeb; }}
.ng {{ background: #fef2f2; }}
.unknown {{ background: #f8fafc; }}
</style>
</head>
<body>
<h1>GeoFEAS package inventory</h1>
<p>package: {html.escape(str(summary.get("package_dir", "")))}</p>
<p>file_count={int(summary.get("file_count", 0) or 0)},
open_supported={int(summary.get("open_supported_count", 0) or 0)},
converter_required={int(summary.get("converter_required_count", 0) or 0)},
blocked={int(summary.get("blocked_count", 0) or 0)},
unknown={int(summary.get("unknown_count", 0) or 0)}</p>
<p>native_private_roundtrip={bool(summary.get("native_private_roundtrip", False))}</p>
<table>
<thead><tr><th>file</th><th>ext</th><th>kind</th><th>status</th><th>open substitute</th><th>blocked reason</th><th>recommended action</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def _workflow_log_verification_html(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for row in summary.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        ok = bool(row.get("passed", False))
        rows.append(
            "<tr class='{cls}'><td>{step}</td><td>{status}</td><td>{etab}</td><td>{atab}</td>"
            "<td>{score:.3f}</td><td>{expected}</td><td>{actual}</td></tr>".format(
                cls="ok" if ok else "ng",
                step=html.escape(str(row.get("expected_step", ""))),
                status=html.escape(str(row.get("status", ""))),
                etab=html.escape(str(row.get("expected_tab", ""))),
                atab=html.escape(str(row.get("actual_tab", ""))),
                score=float(row.get("action_token_overlap", 0.0) or 0.0),
                expected=html.escape(str(row.get("expected_action", ""))),
                actual=html.escape(str(row.get("actual_action", ""))),
            )
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS public workflow log verification</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.ng {{ background: #fef2f2; }}
</style>
</head>
<body>
<h1>GeoFEAS public workflow log verification</h1>
<p>workflow={html.escape(str(summary.get("workflow", "")))}, actual_log={html.escape(str(summary.get("actual_log", "")))}</p>
<p>expected_count={int(summary.get("expected_count", 0) or 0)},
actual_count={int(summary.get("actual_count", 0) or 0)},
matched_count={int(summary.get("matched_count", 0) or 0)},
failed_count={int(summary.get("failed_count", 0) or 0)},
extra_count={int(summary.get("extra_count", 0) or 0)},
passed={bool(summary.get("passed", False))}</p>
<p>{html.escape(str(summary.get("remaining_gap", "")))}</p>
<table>
<thead><tr><th>step</th><th>status</th><th>expected tab</th><th>actual tab</th><th>token overlap</th><th>expected action</th><th>actual action</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def _material_audit_html(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for row in summary.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        severity = str(row.get("severity", "INFO")).lower()
        rows.append(
            "<tr class='{cls}'><td>{material}</td><td>{model}</td><td>{category}</td><td>{status}</td><td>{severity}</td>"
            "<td>{missing}</td><td>{gap}</td><td>{action}</td></tr>".format(
                cls=html.escape(severity),
                material=html.escape(str(row.get("material", ""))),
                model=html.escape(str(row.get("model", ""))),
                category=html.escape(str(row.get("category", ""))),
                status=html.escape(str(row.get("public_status", ""))),
                severity=html.escape(str(row.get("severity", ""))),
                missing=html.escape(str(row.get("missing_required", ""))),
                gap=html.escape(str(row.get("geofeas_private_gap", ""))),
                action=html.escape(str(row.get("recommended_action", ""))),
            )
        )
    warnings = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in summary.get("warnings", []))
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS public material audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
th {{ background: #e5e7eb; }}
.info {{ background: #ecfdf5; }}
.warn {{ background: #fffbeb; }}
.error {{ background: #fef2f2; }}
</style>
</head>
<body>
<h1>GeoFEAS public material audit</h1>
<p>material_count={int(summary.get("material_count", 0) or 0)},
error_count={int(summary.get("error_count", 0) or 0)},
warning_count={int(summary.get("warning_count", 0) or 0)},
public_substitute_count={int(summary.get("public_substitute_count", 0) or 0)},
open_supported_count={int(summary.get("open_supported_count", 0) or 0)},
passed={bool(summary.get("passed", False))}</p>
<p>{html.escape(str(summary.get("remaining_gap", "")))}</p>
<h2>Warnings</h2>
<ul>{warnings}</ul>
<table>
<thead><tr><th>material</th><th>model</th><th>category</th><th>status</th><th>severity</th><th>missing required</th><th>GeoFEAS private gap</th><th>recommended action</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def _post_report_audit_html(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for row in summary.get("checks", []):
        if not isinstance(row, Mapping):
            continue
        severity = str(row.get("severity", "INFO")).lower()
        passed = bool(row.get("passed", False))
        cls = "ok" if passed else severity
        rows.append(
            "<tr class='{cls}'><td>{id}</td><td>{severity}</td><td>{passed}</td><td>{actual}</td><td>{expected}</td><td>{detail}</td></tr>".format(
                cls=html.escape(cls),
                id=html.escape(str(row.get("id", ""))),
                severity=html.escape(str(row.get("severity", ""))),
                passed="OK" if passed else "NG",
                actual=html.escape(str(row.get("actual", ""))),
                expected=html.escape(str(row.get("expected", ""))),
                detail=html.escape(str(row.get("detail", ""))),
            )
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS Post/report audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.info {{ background: #eff6ff; }}
.warn {{ background: #fffbeb; }}
.error {{ background: #fef2f2; }}
</style>
</head>
<body>
<h1>GeoFEAS Post/report audit</h1>
<p>result_dir={html.escape(str(summary.get("result_dir", "")))}</p>
<p>svg_count={int(summary.get("svg_count", 0) or 0)},
png_count={int(summary.get("png_count", 0) or 0)},
value_csv_count={int(summary.get("value_csv_count", 0) or 0)},
post_html_count={int(summary.get("post_html_count", 0) or 0)},
error_count={int(summary.get("error_count", 0) or 0)},
warning_count={int(summary.get("warning_count", 0) or 0)},
passed={bool(summary.get("passed", False))}</p>
<p>{html.escape(str(summary.get("remaining_gap", "")))}</p>
<table>
<thead><tr><th>check</th><th>severity</th><th>judgement</th><th>actual</th><th>expected</th><th>detail</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def _external_product_version_audit_html(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for row in summary.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("public_status", "unknown"))
        cls = {
            "open_supported": "ok",
            "open_diagnostic": "diag",
            "converter_required": "warn",
            "blocked_proprietary": "ng",
        }.get(status, "unknown")
        rows.append(
            "<tr class='{cls}'><td>{path}</td><td>{product}</td><td>{version}</td><td>{vstatus}</td>"
            "<td>{status}</td><td>{field}</td><td>{unsupported}</td><td>{reason}</td><td>{action}</td></tr>".format(
                cls=html.escape(cls),
                path=html.escape(str(row.get("relative_path", ""))),
                product=html.escape(str(row.get("detected_product", ""))),
                version=html.escape(str(row.get("detected_version", ""))),
                vstatus=html.escape(str(row.get("version_status", ""))),
                status=html.escape(status),
                field=html.escape(str(row.get("field_mapping_status", ""))),
                unsupported=html.escape(str(row.get("unsupported_headers", ""))),
                reason=html.escape(str(row.get("blocked_reason", ""))),
                action=html.escape(str(row.get("recommended_action", ""))),
            )
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>External product version audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.diag {{ background: #eff6ff; }}
.warn {{ background: #fffbeb; }}
.ng {{ background: #fef2f2; }}
.unknown {{ background: #f8fafc; }}
</style>
</head>
<body>
<h1>External product version audit</h1>
<p>package={html.escape(str(summary.get("package_dir", "")))}</p>
<p>file_count={int(summary.get("file_count", 0) or 0)},
open_supported={int(summary.get("open_supported_count", 0) or 0)},
converter_required={int(summary.get("converter_required_count", 0) or 0)},
blocked={int(summary.get("blocked_count", 0) or 0)},
unknown={int(summary.get("unknown_count", 0) or 0)},
version_warnings={int(summary.get("version_warning_count", 0) or 0)}</p>
<p>{html.escape(str(summary.get("remaining_gap", "")))}</p>
<table>
<thead><tr><th>file</th><th>product</th><th>version</th><th>version status</th><th>public status</th><th>field mapping</th><th>unsupported headers</th><th>blocked reason</th><th>recommended action</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def _read_delimited_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    return [dict(row) for row in csv.DictReader(text.splitlines(), dialect=dialect)]


def _guess_external_product(path: Path, headers: Iterable[str]) -> str:
    joined = " ".join([path.name, str(path.parent.name), *[str(header) for header in headers]]).lower()
    for product, profile in EXTERNAL_PRODUCT_VERSION_PROFILES.items():
        for alias in profile.get("aliases", ()):
            if str(alias).lower() in joined:
                return product
    if path.suffix.lower() in {".csv", ".tsv"}:
        return _guess_seepage_product(path, [{header: "" for header in headers}], "auto")
    return "generic_csv" if path.suffix.lower() in {".csv", ".tsv"} else "unknown"


def _external_version_status(version: str, profile: Mapping[str, Any]) -> str:
    versions = {str(item).lower() for item in profile.get("public_versions", ())}
    normalized = str(version or "auto").lower()
    if normalized in versions or "auto" in versions:
        return "public_profile_available"
    return "version_unverified"


def _external_file_public_status(ext: str, package_profile: Mapping[str, str], product_profile: Mapping[str, Any]) -> str:
    if ext in set(product_profile.get("private_extensions", ())):
        return "blocked_proprietary"
    if ext in set(product_profile.get("open_extensions", ())):
        return str(package_profile.get("public_status", "open_supported"))
    if package_profile.get("public_status"):
        return str(package_profile["public_status"])
    return "unknown"


def _normalize_external_header(header: Any) -> str:
    mapped = _normalize_seepage_header(header)
    if mapped:
        return mapped
    text = str(header or "").strip().lower().replace(" ", "_")
    aliases = {
        "ux": {"ux", "u_x", "x_displacement", "水平変位"},
        "uy": {"uy", "u_y", "y_displacement", "vertical_displacement", "鉛直変位"},
        "stress": {"stress", "sigma", "応力"},
        "strain": {"strain", "epsilon", "ひずみ"},
        "fl": {"fl", "safety_factor", "liquefaction_safety_factor", "液状化安全率"},
        "ru": {"ru", "excess_pore_pressure_ratio", "過剰間隙水圧比"},
        "element_id": {"element_id", "element", "elem", "要素", "要素番号"},
        "stage": {"stage", "step", "case", "ステージ", "ケース"},
    }
    compact = text.replace("_", "").replace("-", "")
    for normalized, options in aliases.items():
        if text in options or compact in {option.lower().replace("_", "").replace("-", "") for option in options}:
            return normalized
    return ""


def _external_field_mapping_status(
    headers: list[str],
    mapped_headers: Mapping[str, str],
    unsupported_headers: list[str],
    error: str,
) -> str:
    if error:
        return "read_error"
    if not headers:
        return "not_delimited_or_empty"
    if not unsupported_headers:
        return "all_headers_mapped"
    mapped_count = sum(1 for value in mapped_headers.values() if value)
    if mapped_count:
        return "partial_headers_mapped"
    return "unmapped_headers"


def _external_product_blocked_reason(ext: str, public_status: str, version_status: str) -> str:
    reasons: list[str] = []
    if public_status == "blocked_proprietary":
        reasons.append(f"{ext or 'unknown'} is private/undocumented for native roundtrip.")
    elif public_status == "converter_required":
        reasons.append("External converter and version-specific validation are required.")
    elif public_status == "unknown":
        reasons.append("No public compatibility profile is registered for this file type.")
    if version_status != "public_profile_available":
        reasons.append("Detected version is not covered by a public profile.")
    return " ".join(reasons)


def _read_operation_log(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, Mapping)]
        if isinstance(data, Mapping):
            for key in ("operation_log", "operations", "steps", "rows"):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, Mapping)]
        return []
    rows = _read_delimited_rows(path)
    return [row for row in rows if isinstance(row, Mapping)]


def _add_file_check(checks: list[dict[str, Any]], path: Path, check_id: str, severity: str, label: str) -> None:
    exists = path.exists() and path.is_file()
    detail = f"{label}: {path.name}"
    if exists and path.suffix.lower() == ".pdf":
        try:
            exists = path.read_bytes()[:5] == b"%PDF-"
            detail += " / PDF header checked"
        except OSError:
            exists = False
    checks.append(
        {
            "id": check_id,
            "severity": severity,
            "passed": exists,
            "actual": "present" if exists else "missing",
            "expected": "present",
            "detail": detail,
        }
    )


def _add_count_check(checks: list[dict[str, Any]], check_id: str, actual: int, expected_min: int, severity: str, label: str) -> None:
    checks.append(
        {
            "id": check_id,
            "severity": severity,
            "passed": actual >= expected_min,
            "actual": actual,
            "expected": f">={expected_min}",
            "detail": label,
        }
    )


def _read_json_if_exists(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, Mapping) else {}


def _public_post_value_csv_names() -> set[str]:
    return {
        "displacements.csv",
        "reactions.csv",
        "element_results.csv",
        "integration_point_stress.csv",
        "post_case_comparison.csv",
        "load_combinations.csv",
        "liquefaction_state.csv",
        "liquefaction_history.csv",
        "pore_pressure.csv",
        "structural_state.csv",
        "structural_section_forces.csv",
        "interface_state.csv",
        "dynamic_history.csv",
    }


def _normalize_operation_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "step": row.get("step", row.get("index", index)),
        "time": str(row.get("time", row.get("timestamp", "")) or ""),
        "tab": str(row.get("tab", row.get("mode", row.get("screen", ""))) or "").strip(),
        "action": str(row.get("action", row.get("command", row.get("operation", row.get("text", "")))) or "").strip(),
        "expected": str(row.get("expected", row.get("result", "")) or "").strip(),
    }


def _operation_token_overlap(expected: str, actual: str) -> float:
    expected_tokens = _operation_tokens(expected)
    actual_tokens = _operation_tokens(actual)
    if not expected_tokens:
        return 1.0 if not actual_tokens else 0.0
    if not actual_tokens:
        return 0.0
    return len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)


def _operation_tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
    stop = {"a", "an", "the", "and", "or", "with", "to", "by", "after", "before", "of", "for", "public", "profile"}
    return {token for token in normalized.split() if len(token) >= 3 and token not in stop}


def _casefold(value: Any) -> str:
    return str(value or "").strip().casefold()


def _flatten_material_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(str(key), nested)
                if prefix:
                    visit(f"{prefix}.{key}", nested)
        else:
            out[prefix] = value

    for key, value in spec.items():
        visit(str(key), value)
    return out


def _has_any_alias(fields: Mapping[str, Any], aliases: Iterable[str]) -> bool:
    lowered = {str(key).lower(): value for key, value in fields.items()}
    for alias in aliases:
        text = str(alias).lower()
        if text in lowered and lowered[text] not in (None, ""):
            return True
        if f"liquefaction.{text}" in lowered and lowered[f"liquefaction.{text}"] not in (None, ""):
            return True
        if f"river_seismic_guideline.{text}" in lowered and lowered[f"river_seismic_guideline.{text}"] not in (None, ""):
            return True
    return False


def _alias_group_label(group: Iterable[str]) -> str:
    return "/".join(str(item) for item in group)


def _material_private_gap(model: str, status: str) -> str:
    if status == "open_supported":
        if model in {"mohr_coulomb", "mc"}:
            return "Exact GeoFEAS Mohr-Coulomb active-set details require reference comparison."
        return ""
    if model in {"liquefaction", "bilinear_liquefaction"}:
        return "Exact ru generation/dissipation, FL/RL correlations, hidden defaults, and cyclic history equations are not public."
    if model in {"uw_clay", "pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay"}:
        return "Exact internal variables, hardening law, dilatancy law, and product-version-specific integration rules are not public."
    if status == "public_substitute":
        return "Public curve/substitute behavior is available, but exact GeoFEAS product internals are not asserted."
    return "No public GeoFEAS material profile is available."


def _material_stage_workflows(config: Mapping[str, Any]) -> list[str]:
    raw = config.get("stages", config.get("steps", []))
    workflows: list[str] = []
    if isinstance(raw, list):
        for stage in raw:
            if isinstance(stage, Mapping) and stage.get("geofeas_workflow"):
                workflows.append(str(stage.get("geofeas_workflow")))
    return sorted(set(workflows))


def _normalize_seepage_header(header: Any) -> str:
    text = str(header or "").strip().replace(" ", "_")
    lower = text.lower()
    for normalized, aliases in _SEEPAGE_HEADER_ALIASES.items():
        if text in aliases or lower in {alias.lower() for alias in aliases}:
            return normalized
    compact = lower.replace("_", "").replace("-", "")
    for normalized, aliases in _SEEPAGE_HEADER_ALIASES.items():
        if compact in {alias.lower().replace("_", "").replace("-", "") for alias in aliases}:
            return normalized
    return ""


def _guess_seepage_product(path: Path, rows: list[dict[str, str]], product: str) -> str:
    if product and product != "auto":
        return product
    joined = " ".join([path.name, *list(rows[0].keys() if rows else {})]).lower()
    if "uc-1" in joined or "uc1" in joined:
        return "UC-1"
    if "geofeas" in joined or "geo feas" in joined:
        return "GeoFEAS"
    if "vgflow" in joined:
        return "VGFlow"
    return "generic_csv"


def _detect_external_version(path: Path, rows: list[dict[str, str]]) -> str:
    joined = " ".join([path.name, *list(rows[0].keys() if rows else {}), *list((rows[0].values() if rows else []))]).lower()
    for marker in ("ver.", "version", "v"):
        idx = joined.find(marker)
        if idx >= 0:
            tail = joined[idx + len(marker) :].strip(" _:-")
            token = tail.split()[0].strip(",;") if tail else ""
            if token:
                return token
    return "auto"


def _binary_hint(path: Path) -> str:
    try:
        sample = path.read_bytes()[:1024]
    except OSError:
        return "unreadable"
    if b"\x00" in sample:
        return "binary"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return "binary_or_legacy_text"
    return "text"


def _package_blocked_reason(suffix: str, status: str) -> str:
    if status == "blocked_proprietary":
        return f"{suffix or 'unknown'} native format is not publicly specified for safe roundtrip."
    if status == "converter_required":
        return "Native read/write depends on an external converter and version-specific validation."
    if status == "unknown":
        return "No public GeoFEM compatibility profile is registered for this extension."
    return ""


def _seepage_key(row: Mapping[str, Any]) -> str:
    return f"{float(row.get('time', 0.0) or 0.0):.12g}|{row.get('node_id', '')}"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _guess_key_fields(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> tuple[str, ...]:
    fields = set(actual[0] if actual else {}) & set(reference[0] if reference else {})
    for candidate in (
        ("element_id", "x"),
        ("element_id", "gp"),
        ("element_id",),
        ("interface_id", "gp"),
        ("interface_id",),
        ("combination", "case"),
        ("node_id", "dof"),
        ("node_id",),
        ("stage",),
        ("id",),
    ):
        if all(field in fields for field in candidate):
            return candidate
    return ()


def _guess_value_fields(actual: list[dict[str, str]], reference: list[dict[str, str]], keys: list[str]) -> tuple[str, ...]:
    fields = [field for field in (actual[0] if actual else {}) if field in (reference[0] if reference else {}) and field not in keys]
    numeric: list[str] = []
    for field in fields:
        values = [row.get(field, "") for row in actual[:5] + reference[:5]]
        if values and all(_is_float(value) for value in values if value != ""):
            numeric.append(field)
    return tuple(numeric)


def _row_key(row: Mapping[str, Any], keys: list[str]) -> str:
    return "|".join(str(row.get(key, "")) for key in keys)


def _is_float(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


__all__ = [
    "EXTERNAL_SEEPAGE_PRODUCT_PROFILES",
    "EXTERNAL_PRODUCT_VERSION_PROFILES",
    "GEOFEAS_DYNAMIC_COMPARISON_SPECS",
    "GEOFEAS_PACKAGE_FILE_PROFILES",
    "GEOFEAS_PUBLIC_MATERIAL_PROFILES",
    "GEOFEAS_STRUCTURAL_COMPARISON_SPECS",
    "audit_public_material_profile",
    "audit_post_report_package",
    "audit_external_product_versions",
    "compare_external_seepage_roundtrip",
    "compare_external_seepage_results",
    "compare_geofeas_dynamic_sample",
    "compare_geofeas_reference_csv",
    "compare_geofeas_stage_package",
    "diagnose_external_seepage_version",
    "diagnose_geofeas_package_files",
    "import_external_seepage_results",
    "write_external_seepage_results",
    "write_geofeas_package_inventory",
    "write_geofeas_comparison_csv",
    "write_geofeas_package_report",
    "write_geofeas_tolerance_report",
    "write_public_material_audit",
    "write_post_report_audit",
    "write_external_product_version_audit",
    "verify_public_workflow_log",
    "write_public_workflow_log_verification",
]
