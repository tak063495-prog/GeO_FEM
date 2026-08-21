from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.geofeas_verification import audit_external_product_versions


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description="Audit external GeoFEAS/UC-1/VGFlow/UWLC/DWG/open-file version and header compatibility.",
    )
    parser.add_argument("--package", required=True, type=Path, help="Folder containing external product files.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for audit JSON/CSV/HTML.")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Return non-zero when private/proprietary files are present.")
    parser.add_argument("--fail-on-version-warning", action="store_true", help="Return non-zero when version/profile warnings remain.")
    args = parser.parse_args(argv)

    summary = audit_external_product_versions(args.package, output_dir=args.out)
    compact = {
        "file_count": int(summary.get("file_count", 0) or 0),
        "open_supported_count": int(summary.get("open_supported_count", 0) or 0),
        "converter_required_count": int(summary.get("converter_required_count", 0) or 0),
        "blocked_count": int(summary.get("blocked_count", 0) or 0),
        "unknown_count": int(summary.get("unknown_count", 0) or 0),
        "version_warning_count": int(summary.get("version_warning_count", 0) or 0),
        "exact_product_version_parity": bool(summary.get("exact_product_version_parity", False)),
        "json": str(args.out / "external_product_version_audit.json"),
        "html": str(args.out / "external_product_version_audit.html"),
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    if args.fail_on_blocked and compact["blocked_count"] > 0:
        return 1
    if args.fail_on_version_warning and compact["version_warning_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
