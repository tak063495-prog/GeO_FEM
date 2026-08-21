from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.geofeas_verification import verify_public_workflow_log


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description="Verify an observed GUI operation log against a public GeoFEAS 2D workflow template.",
    )
    parser.add_argument("--workflow", default="tunnel_excavation", help="Public workflow id, e.g. tunnel_excavation.")
    parser.add_argument("--log", required=True, type=Path, help="Observed operation log JSON/CSV.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for verification JSON/CSV/HTML.")
    parser.add_argument("--min-token-overlap", type=float, default=0.35, help="Required action token overlap per step.")
    parser.add_argument("--allow-partial", action="store_true", help="Return success even when steps are missing/mismatched.")
    args = parser.parse_args(argv)

    summary = verify_public_workflow_log(
        args.workflow,
        args.log,
        output_dir=args.out,
        min_action_token_overlap=args.min_token_overlap,
    )
    compact = {
        "passed": bool(summary.get("passed", False)),
        "workflow": summary.get("workflow"),
        "expected_count": int(summary.get("expected_count", 0) or 0),
        "actual_count": int(summary.get("actual_count", 0) or 0),
        "matched_count": int(summary.get("matched_count", 0) or 0),
        "failed_count": int(summary.get("failed_count", 0) or 0),
        "extra_count": int(summary.get("extra_count", 0) or 0),
        "json": str(args.out / "geofeas_public_workflow_log_verification.json"),
        "html": str(args.out / "geofeas_public_workflow_log_verification.html"),
    }
    if emit:
        print(json.dumps(compact, ensure_ascii=False))
    return 0 if summary.get("passed", False) or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
