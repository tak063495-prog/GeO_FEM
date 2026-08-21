from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.geofeas_verification import diagnose_geofeas_package_files


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory a GeoFEAS/GeoFEM exchange folder and classify open substitutes versus blocked private formats.",
    )
    parser.add_argument("--package", required=True, type=Path, help="Folder containing GeoFEAS/GeoFEM/CAD/result files.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for inventory JSON/CSV/HTML.")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Return non-zero when private/proprietary files are present.")
    args = parser.parse_args(argv)

    summary = diagnose_geofeas_package_files(args.package, output_dir=args.out)
    compact = {
        "file_count": int(summary.get("file_count", 0) or 0),
        "open_supported_count": int(summary.get("open_supported_count", 0) or 0),
        "converter_required_count": int(summary.get("converter_required_count", 0) or 0),
        "blocked_count": int(summary.get("blocked_count", 0) or 0),
        "unknown_count": int(summary.get("unknown_count", 0) or 0),
        "native_private_roundtrip": bool(summary.get("native_private_roundtrip", False)),
        "json": str(args.out / "geofeas_package_inventory.json"),
        "html": str(args.out / "geofeas_package_inventory.html"),
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    if args.fail_on_blocked and compact["blocked_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
