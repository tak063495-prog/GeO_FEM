from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.geofeas_verification import audit_post_report_package


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description="Audit GeoFEM Post/report artifacts against the public GeoFEAS 2D output profile.",
    )
    parser.add_argument("--result", required=True, type=Path, help="GeoFEM result/output directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for Post/report audit JSON/CSV/HTML.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional accepted visual baseline directory.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero when required public Post/report artifacts are missing.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return non-zero when warnings remain, including missing visual baselines.")
    args = parser.parse_args(argv)

    summary = audit_post_report_package(args.result, baseline_dir=args.baseline, output_dir=args.out)
    compact = {
        "passed": bool(summary.get("passed", False)),
        "error_count": int(summary.get("error_count", 0) or 0),
        "warning_count": int(summary.get("warning_count", 0) or 0),
        "svg_count": int(summary.get("svg_count", 0) or 0),
        "value_csv_count": int(summary.get("value_csv_count", 0) or 0),
        "pixel_equivalent_post_claim": bool(summary.get("pixel_equivalent_post_claim", False)),
        "native_oss_roundtrip_claim": bool(summary.get("native_oss_roundtrip_claim", False)),
        "json": str(args.out / "geofeas_post_report_audit.json"),
        "html": str(args.out / "geofeas_post_report_audit.html"),
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    if args.fail_on_error and not summary.get("passed", False):
        return 1
    if args.fail_on_warning and int(summary.get("warning_count", 0) or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
