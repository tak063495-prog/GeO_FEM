from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.verification_benchmarks import run_geofeas_benchmark_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GeoFEAS-like GeoFEM 2D verification benchmarks.")
    parser.add_argument("--out", default="benchmark_results", help="Output directory for benchmark artifacts.")
    args = parser.parse_args(argv)
    summary = run_geofeas_benchmark_suite(Path(args.out))
    public = summary.get("public_compatibility", {})
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "case_count": summary["case_count"],
                "failed_count": summary["failed_count"],
                "out": str(Path(args.out)),
                "geofeas_public_compatibility_matrix": public.get("json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
