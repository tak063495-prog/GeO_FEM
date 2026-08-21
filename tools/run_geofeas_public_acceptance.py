from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.geofeas_verification import (
    GEOFEAS_DYNAMIC_COMPARISON_SPECS,
    GEOFEAS_STRUCTURAL_COMPARISON_SPECS,
    audit_external_product_versions,
    audit_post_report_package,
    audit_public_material_profile,
    compare_geofeas_dynamic_sample,
    compare_geofeas_stage_package,
    diagnose_geofeas_package_files,
    verify_public_workflow_log,
)


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".json", ".jsn"}:
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("config must contain a mapping at the top level")
    return data


def _module_row(name: str, kind: str, summary: Mapping[str, Any] | None, artifact_dir: Path, *, skipped_reason: str = "") -> dict[str, Any]:
    if summary is None:
        return {
            "module": name,
            "kind": kind,
            "status": "skipped",
            "passed": True,
            "error_count": 0,
            "warning_count": 0,
            "blocked_count": 0,
            "artifact_dir": "",
            "detail": skipped_reason,
        }
    error_count = int(summary.get("error_count", summary.get("failed_count", 0)) or 0)
    warning_count = int(summary.get("warning_count", summary.get("version_warning_count", 0)) or 0)
    blocked_count = int(summary.get("blocked_count", 0) or 0)
    passed = bool(summary.get("passed", error_count == 0))
    return {
        "module": name,
        "kind": kind,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "error_count": error_count,
        "warning_count": warning_count,
        "blocked_count": blocked_count,
        "artifact_dir": str(artifact_dir),
        "detail": str(summary.get("remaining_gap", summary.get("blocked_reason", ""))),
    }


def _write_acceptance_outputs(summary: Mapping[str, Any], out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "geofeas_public_acceptance_summary.json"
    csv_path = out / "geofeas_public_acceptance_summary.csv"
    html_path = out / "geofeas_public_acceptance_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fieldnames = ["module", "kind", "status", "passed", "error_count", "warning_count", "blocked_count", "artifact_dir", "detail"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.get("modules", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    html_path.write_text(_acceptance_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _acceptance_html(summary: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for row in summary.get("modules", []):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", "skipped"))
        cls = "ok" if status == "passed" else ("skip" if status == "skipped" else "ng")
        rows.append(
            "<tr class='{cls}'><td>{module}</td><td>{kind}</td><td>{status}</td><td>{errors}</td>"
            "<td>{warnings}</td><td>{blocked}</td><td>{artifacts}</td><td>{detail}</td></tr>".format(
                cls=cls,
                module=html.escape(str(row.get("module", ""))),
                kind=html.escape(str(row.get("kind", ""))),
                status=html.escape(status),
                errors=int(row.get("error_count", 0) or 0),
                warnings=int(row.get("warning_count", 0) or 0),
                blocked=int(row.get("blocked_count", 0) or 0),
                artifacts=html.escape(str(row.get("artifact_dir", ""))),
                detail=html.escape(str(row.get("detail", ""))),
            )
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>GeoFEAS public acceptance gate</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
th {{ background: #e5e7eb; }}
.ok {{ background: #ecfdf5; }}
.skip {{ background: #f8fafc; }}
.ng {{ background: #fef2f2; }}
</style>
</head>
<body>
<h1>GeoFEAS public acceptance gate</h1>
<p>passed={bool(summary.get("passed", False))},
module_count={int(summary.get("module_count", 0) or 0)},
skipped_count={int(summary.get("skipped_count", 0) or 0)},
error_count={int(summary.get("error_count", 0) or 0)},
warning_count={int(summary.get("warning_count", 0) or 0)},
blocked_count={int(summary.get("blocked_count", 0) or 0)}</p>
<p>{html.escape(str(summary.get("remaining_gap", "")))}</p>
<table>
<thead><tr><th>module</th><th>kind</th><th>status</th><th>errors</th><th>warnings</th><th>blocked</th><th>artifacts</th><th>detail</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(description="Run the GeoFEAS 2D public-profile acceptance gate.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for consolidated acceptance artifacts.")
    parser.add_argument("--config", type=Path, default=None, help="GeoFEM YAML/JSON input for material audit.")
    parser.add_argument("--result", type=Path, default=None, help="GeoFEM result directory for Post/report audit.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional visual baseline directory for Post/report audit.")
    parser.add_argument("--package", type=Path, default=None, help="Mixed GeoFEAS/CAD/external product package directory.")
    parser.add_argument("--operation-log", type=Path, default=None, help="Observed GUI operation log JSON/CSV.")
    parser.add_argument("--workflow", default="tunnel_excavation", help="Public workflow id for operation-log verification.")
    parser.add_argument("--actual", type=Path, default=None, help="GeoFEM stage output directory for official/reference comparison.")
    parser.add_argument("--reference", type=Path, default=None, help="GeoFEAS/reference output directory for official/reference comparison.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic comparison specs for official/reference comparison.")
    parser.add_argument("--require-all", action="store_true", help="Fail when any acceptance module is skipped.")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Fail when blocked private/external files are detected.")
    args = parser.parse_args(argv)

    out = args.out
    modules: list[dict[str, Any]] = []

    if args.config is not None:
        material_dir = out / "material_audit"
        material = audit_public_material_profile(_load_config(args.config), output_dir=material_dir)
        modules.append(_module_row("material_profile", "materials", material, material_dir))
    else:
        modules.append(_module_row("material_profile", "materials", None, out, skipped_reason="--config was not supplied"))

    if args.result is not None:
        post_dir = out / "post_report_audit"
        post = audit_post_report_package(args.result, baseline_dir=args.baseline, output_dir=post_dir)
        modules.append(_module_row("post_report", "post/report", post, post_dir))
    else:
        modules.append(_module_row("post_report", "post/report", None, out, skipped_reason="--result was not supplied"))

    if args.package is not None:
        package_dir = out / "package_inventory"
        package = diagnose_geofeas_package_files(args.package, output_dir=package_dir)
        modules.append(_module_row("package_inventory", "files", package, package_dir))
        external_dir = out / "external_product_versions"
        external = audit_external_product_versions(args.package, output_dir=external_dir)
        modules.append(_module_row("external_product_versions", "external products", external, external_dir))
    else:
        modules.append(_module_row("package_inventory", "files", None, out, skipped_reason="--package was not supplied"))
        modules.append(_module_row("external_product_versions", "external products", None, out, skipped_reason="--package was not supplied"))

    if args.operation_log is not None:
        workflow_dir = out / "workflow_log"
        workflow = verify_public_workflow_log(args.workflow, args.operation_log, output_dir=workflow_dir)
        modules.append(_module_row("workflow_log", "GUI workflow", workflow, workflow_dir))
    else:
        modules.append(_module_row("workflow_log", "GUI workflow", None, out, skipped_reason="--operation-log was not supplied"))

    if args.actual is not None and args.reference is not None:
        reference_dir = out / "reference_comparison"
        specs = GEOFEAS_DYNAMIC_COMPARISON_SPECS if args.dynamic else GEOFEAS_STRUCTURAL_COMPARISON_SPECS
        compare = compare_geofeas_dynamic_sample if args.dynamic else compare_geofeas_stage_package
        reference = compare(args.actual, args.reference, output_dir=reference_dir, specs=specs)
        if int(reference.get("compared_count", 0) or 0) == 0:
            reference = {**reference, "passed": False, "failed_count": int(reference.get("failed_count", 0) or 0) + 1, "blocked_reason": "No recognized reference files were compared."}
        modules.append(_module_row("reference_comparison", "official/reference values", reference, reference_dir))
    else:
        modules.append(_module_row("reference_comparison", "official/reference values", None, out, skipped_reason="--actual and --reference were not both supplied"))

    skipped = [row for row in modules if row["status"] == "skipped"]
    failed = [row for row in modules if not bool(row.get("passed", False))]
    blocked_count = sum(int(row.get("blocked_count", 0) or 0) for row in modules)
    require_all_failed = bool(args.require_all and skipped)
    blocked_failed = bool(args.fail_on_blocked and blocked_count)
    passed = not failed and not require_all_failed and not blocked_failed
    summary: dict[str, Any] = {
        "schema": "geofem.geofeas_public_acceptance.v1",
        "passed": passed,
        "module_count": len(modules),
        "skipped_count": len(skipped),
        "error_count": sum(int(row.get("error_count", 0) or 0) for row in modules) + (len(skipped) if require_all_failed else 0),
        "warning_count": sum(int(row.get("warning_count", 0) or 0) for row in modules),
        "blocked_count": blocked_count,
        "require_all": bool(args.require_all),
        "fail_on_blocked": bool(args.fail_on_blocked),
        "native_geo_feas_equivalence_claim": False,
        "remaining_gap": "This gate consolidates public substitutes and diagnostics; exact GeoFEAS commercial equivalence still requires official models, outputs, UI captures, and accepted tolerances.",
        "modules": modules,
    }
    artifacts = _write_acceptance_outputs(summary, out)
    compact = {
        "passed": summary["passed"],
        "module_count": summary["module_count"],
        "skipped_count": summary["skipped_count"],
        "error_count": summary["error_count"],
        "warning_count": summary["warning_count"],
        "blocked_count": summary["blocked_count"],
        "json": artifacts["json"],
        "html": artifacts["html"],
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
