from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.vgflow2d_performance import run_vgflow2d_performance_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run VGFlow 2D public substitute performance benchmarks.")
    parser.add_argument("--out", default="runs/vgflow2d_performance", help="Output directory for CSV/JSON artifacts.")
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke case instead of small/medium/large cases.")
    parser.add_argument("--skip-solve", action="store_true", help="Only benchmark mesh projection plan build/apply paths.")
    args = parser.parse_args(argv)
    payload = run_vgflow2d_performance_benchmark(Path(args.out), quick=args.quick, include_solve=not args.skip_solve)
    print(payload["artifacts"]["json"])
    print(payload["artifacts"]["csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
