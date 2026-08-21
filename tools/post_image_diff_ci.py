from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geofem_app.post_image_diff import compare_images, create_sample_post_image


def _load_matrix(path: Path) -> list[dict[str, object]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = data.get("post_image_diff_cases", data.get("cases", data))
    if not isinstance(cases, list):
        raise ValueError(f"matrix must contain a list of cases: {path}")
    return [dict(case) for case in cases if isinstance(case, dict)]


def _run_case(case: dict[str, object], default_out_dir: Path) -> dict[str, object]:
    name = str(case.get("name", f"case_{len(str(case))}")).replace(" ", "_")
    threshold = float(case.get("threshold", 0.02))
    baseline = Path(str(case["baseline"]))
    current = Path(str(case.get("current", default_out_dir / f"post_current_{name}.png")))
    if bool(case.get("generate_sample", True)):
        create_sample_post_image(current)
    elif not current.exists():
        raise FileNotFoundError(f"current image does not exist: {current}")
    if not baseline.exists():
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, baseline)
        result: dict[str, object] = {"ok": True, "baseline_created": True}
    else:
        result = compare_images(current, baseline, threshold=threshold)
        result["baseline_created"] = False
    result.update({"name": name, "current": str(current), "baseline": str(baseline), "threshold": threshold})
    return result


def main(argv: list[str] | None = None, *, emit: bool = True) -> int:
    parser = argparse.ArgumentParser(description="Generate and compare GeoFEM Post images for CI.")
    parser.add_argument("--current", type=Path, default=None, help="Current Post PNG. Created when --generate-sample is set.")
    parser.add_argument("--baseline", type=Path, default=Path("post_baselines/post_baseline.png"), help="Baseline Post PNG.")
    parser.add_argument("--out", type=Path, default=Path("post_image_diff_result.json"), help="JSON result path.")
    parser.add_argument("--threshold", type=float, default=0.02, help="Maximum allowed changed-pixel ratio.")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a deterministic sample Post image before comparing.")
    parser.add_argument("--matrix", type=Path, default=None, help="YAML file with multiple Post image diff cases.")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.matrix is not None:
        cases = _load_matrix(args.matrix)
        results = [_run_case(case, args.out.parent) for case in cases]
        result: dict[str, object] = {
            "ok": all(bool(item.get("ok")) for item in results),
            "matrix": str(args.matrix),
            "cases": results,
            "case_count": len(results),
        }
    else:
        current = args.current or args.out.with_name("post_current.png")
        if args.generate_sample:
            create_sample_post_image(current)
        elif not current.exists():
            parser.error(f"current image does not exist: {current}")

        if not args.baseline.exists():
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, args.baseline)
            result = {"ok": True, "baseline_created": True, "current": str(current), "baseline": str(args.baseline)}
        else:
            result = compare_images(current, args.baseline, threshold=args.threshold)
            result.update({"baseline_created": False, "current": str(current), "baseline": str(args.baseline)})

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if emit:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
