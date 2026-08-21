from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.geofeas_verification import audit_public_material_profile


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


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description="Audit GeoFEM material definitions against the public GeoFEAS 2D substitute profile.",
    )
    parser.add_argument("--config", required=True, type=Path, help="GeoFEM YAML/JSON input file.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for material audit JSON/CSV/HTML.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero when unknown models or missing required fields are found.")
    args = parser.parse_args(argv)

    summary = audit_public_material_profile(_load_config(args.config), output_dir=args.out)
    compact = {
        "passed": bool(summary.get("passed", False)),
        "material_count": int(summary.get("material_count", 0) or 0),
        "error_count": int(summary.get("error_count", 0) or 0),
        "warning_count": int(summary.get("warning_count", 0) or 0),
        "public_substitute_count": int(summary.get("public_substitute_count", 0) or 0),
        "open_supported_count": int(summary.get("open_supported_count", 0) or 0),
        "json": str(args.out / "geofeas_public_material_audit.json"),
        "html": str(args.out / "geofeas_public_material_audit.html"),
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    return 1 if args.fail_on_error and not summary.get("passed", False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
