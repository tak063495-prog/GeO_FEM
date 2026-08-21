from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


EVIDENCE_ITEMS: tuple[dict[str, Any], ...] = (
    {
        "id": "E1",
        "gap": "Official GeoFEAS numerical equivalence",
        "needed": "GeoFEAS 2D official/sample result package with model, result CSV/tables, units, version, and accepted tolerances.",
        "accepted_files": "*.csv, *.tsv, exported tables, report PDFs, screenshots",
        "geo_fem_tool": "tools/compare_geofeas_reference_package.py",
        "acceptance_output": "reference_comparison/geofeas_reference_package_summary.json",
        "blocked_until": "Official result values or exported reference tables are supplied.",
    },
    {
        "id": "E2",
        "gap": "Proprietary file roundtrip compatibility",
        "needed": "Representative *.GF2/*.sta/*.msh/*.oss files, format specification or converter-validated expected outputs.",
        "accepted_files": "*.GF2, *.sta, *.msh, *.oss, converted DXF/SXF/CSV packages",
        "geo_fem_tool": "tools/diagnose_geofeas_package.py",
        "acceptance_output": "package_inventory/geofeas_package_inventory.json",
        "blocked_until": "Private formats are documented or equivalent open exports are supplied.",
    },
    {
        "id": "E3",
        "gap": "Commercial GUI identity",
        "needed": "Licensed-product operation logs or screen captures with command order, dialogs, selection states, and expected wording.",
        "accepted_files": "operation_log.json/csv, screen captures, workflow notes",
        "geo_fem_tool": "tools/verify_geofeas_workflow_log.py",
        "acceptance_output": "workflow_log/geofeas_public_workflow_log_verification.json",
        "blocked_until": "GeoFEAS UI captures/logs and accepted tolerances are supplied.",
    },
    {
        "id": "E4",
        "gap": "Non-public constitutive and liquefaction internals",
        "needed": "Material-model reference results, parameter sets, test curves, hidden defaults, and version-specific behavior notes.",
        "accepted_files": "material YAML/JSON, test-curve CSV, GeoFEAS result tables, formula/version notes",
        "geo_fem_tool": "tools/audit_geofeas_material_profile.py",
        "acceptance_output": "material_audit/geofeas_public_material_audit.json",
        "blocked_until": "Reference material behavior and tolerances are supplied.",
    },
    {
        "id": "E5",
        "gap": "Commercial Post and report visual parity",
        "needed": "Accepted Post baseline images, output-condition files, report templates, legend defaults, and visual tolerances.",
        "accepted_files": "*.png, *.svg, *.pdf, *.html, *.oss, visual-baseline folders",
        "geo_fem_tool": "tools/audit_geofeas_post_report.py",
        "acceptance_output": "post_report_audit/geofeas_post_report_audit.json",
        "blocked_until": "Commercial visual/report baselines and tolerances are supplied.",
    },
    {
        "id": "E6",
        "gap": "External product version parity",
        "needed": "VGFlow/GeoFEAS/UWLC/UC-1 files across target versions plus expected normalized headers and attribute preservation rules.",
        "accepted_files": "*.csv, *.tsv, *.txt, *.dwg, converted DXF/SXF, product-version notes",
        "geo_fem_tool": "tools/audit_external_product_versions.py",
        "acceptance_output": "external_product_versions/external_product_version_audit.json",
        "blocked_until": "Target product files and version-specific expectations are supplied.",
    },
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["id", "gap", "needed", "accepted_files", "geo_fem_tool", "acceptance_output", "blocked_until"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr><td>{id}</td><td>{gap}</td><td>{needed}</td><td>{files}</td><td>{tool}</td><td>{out}</td><td>{blocked}</td></tr>".format(
                id=html.escape(str(row["id"])),
                gap=html.escape(str(row["gap"])),
                needed=html.escape(str(row["needed"])),
                files=html.escape(str(row["accepted_files"])),
                tool=html.escape(str(row["geo_fem_tool"])),
                out=html.escape(str(row["acceptance_output"])),
                blocked=html.escape(str(row["blocked_until"])),
            )
        )
    path.write_text(
        """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS 2D evidence request pack</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; color: #111827; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }
th { background: #e5e7eb; }
</style>
</head>
<body>
<h1>GeoFEAS 2D evidence request pack</h1>
<p>This pack lists the external evidence required before GeoFEM can claim strict commercial GeoFEAS 2D equivalence.</p>
<table>
<thead><tr><th>ID</th><th>Gap</th><th>Needed evidence</th><th>Accepted files</th><th>GeoFEM tool</th><th>Acceptance output</th><th>Blocked until</th></tr></thead>
<tbody>
"""
        + "".join(table_rows)
        + """
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GeoFEAS 2D Evidence Request Pack",
        "",
        "This folder describes the external files needed to move from public-profile compatibility to strict GeoFEAS 2D equivalence claims.",
        "",
        "Run the public acceptance gate after adding evidence:",
        "",
        "```powershell",
        "python tools/run_geofeas_public_acceptance.py --out acceptance --config model.json --result result_dir --package evidence_package --operation-log operation_log.json --actual actual_stage --reference reference_stage --require-all",
        "```",
        "",
        "## Required Evidence",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['id']} {row['gap']}",
                "",
                f"- Needed: {row['needed']}",
                f"- Accepted files: `{row['accepted_files']}`",
                f"- GeoFEM tool: `{row['geo_fem_tool']}`",
                f"- Acceptance output: `{row['acceptance_output']}`",
                f"- Blocked until: {row['blocked_until']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_placeholders(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        item_dir = root / row["id"]
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {row['id']} {row['gap']}",
                    "",
                    f"Needed: {row['needed']}",
                    "",
                    f"Accepted files: `{row['accepted_files']}`",
                    "",
                    f"Validation tool: `{row['geo_fem_tool']}`",
                    "",
                    "Place official/commercial evidence files for this item in this folder.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def create_evidence_request_pack(out: Path) -> dict[str, Any]:
    rows = [dict(row) for row in EVIDENCE_ITEMS]
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "geofeas_evidence_request.csv", rows)
    _write_html(out / "geofeas_evidence_request.html", rows)
    _write_readme(out / "README.md", rows)
    _write_placeholders(out / "evidence", rows)
    manifest = {
        "schema": "geofem.geofeas_evidence_request_pack.v1",
        "item_count": len(rows),
        "native_geo_feas_equivalence_claim": False,
        "acceptance_gate": "tools/run_geofeas_public_acceptance.py",
        "rows": rows,
    }
    (out / "geofeas_evidence_request.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **manifest,
        "json": str(out / "geofeas_evidence_request.json"),
        "csv": str(out / "geofeas_evidence_request.csv"),
        "html": str(out / "geofeas_evidence_request.html"),
        "readme": str(out / "README.md"),
        "evidence_dir": str(out / "evidence"),
    }


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(description="Create a GeoFEAS 2D evidence request pack for unresolved commercial-equivalence gaps.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the evidence request pack.")
    args = parser.parse_args(argv)
    summary = create_evidence_request_pack(args.out)
    compact = {
        "item_count": summary["item_count"],
        "native_geo_feas_equivalence_claim": summary["native_geo_feas_equivalence_claim"],
        "json": summary["json"],
        "html": summary["html"],
        "evidence_dir": summary["evidence_dir"],
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
