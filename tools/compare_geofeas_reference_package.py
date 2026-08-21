from __future__ import annotations

import argparse
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
    compare_geofeas_dynamic_sample,
    compare_geofeas_stage_package,
)


def _override_specs(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    rtol: float | None,
    atol: float | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        item = dict(spec)
        if rtol is not None:
            item["rtol"] = rtol
        if atol is not None:
            item["atol"] = atol
        out[name] = item
    return out


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description="Compare GeoFEM stage outputs with GeoFEAS/reference CSV package outputs.",
    )
    parser.add_argument("--actual", required=True, type=Path, help="GeoFEM stage output directory.")
    parser.add_argument("--reference", required=True, type=Path, help="GeoFEAS/reference output directory.")
    parser.add_argument("--out", required=True, type=Path, help="Directory for comparison JSON/CSV/HTML artifacts.")
    parser.add_argument("--dynamic", action="store_true", help="Include dynamic-history comparison fields.")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance for all compared files.")
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance for all compared files.")
    parser.add_argument(
        "--allow-empty-reference",
        action="store_true",
        help="Return success even when the reference directory contains no recognized comparison files.",
    )
    args = parser.parse_args(argv)

    base_specs = GEOFEAS_DYNAMIC_COMPARISON_SPECS if args.dynamic else GEOFEAS_STRUCTURAL_COMPARISON_SPECS
    specs = _override_specs(base_specs, rtol=args.rtol, atol=args.atol)
    compare = compare_geofeas_dynamic_sample if args.dynamic else compare_geofeas_stage_package
    summary = compare(args.actual, args.reference, output_dir=args.out, specs=specs)
    summary["schema"] = "geofem.geofeas_reference_package_cli.v1"
    summary["mode"] = "dynamic" if args.dynamic else "static"
    summary["recognized_reference_files"] = [name for name in specs if (args.reference / name).exists()]
    summary["allow_empty_reference"] = bool(args.allow_empty_reference)
    if int(summary.get("compared_count", 0) or 0) == 0 and not args.allow_empty_reference:
        summary["passed"] = False
        summary["failed_count"] = int(summary.get("failed_count", 0) or 0) + 1
        summary["empty_reference_package"] = True
        summary["blocked_reason"] = "No recognized GeoFEAS/reference CSV files were found."
    else:
        summary["empty_reference_package"] = False

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "geofeas_reference_package_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    compact = {
        "passed": bool(summary.get("passed", False)),
        "mode": summary["mode"],
        "compared_count": int(summary.get("compared_count", 0) or 0),
        "failed_count": int(summary.get("failed_count", 0) or 0),
        "max_abs_error": float(summary.get("max_abs_error", 0.0) or 0.0),
        "max_rel_error": float(summary.get("max_rel_error", 0.0) or 0.0),
        "summary": str(summary_path),
        "html": str(args.out / "geofeas_package_tolerance.html"),
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    return 0 if summary.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
