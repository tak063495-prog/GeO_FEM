"""Performance benchmark helpers for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .fem2d_mesh import mesh_from_config
from .mesh_coupling import apply_scalar_projection_plan, build_scalar_projection_plan, upgrade_quad4_mesh_to_quad8
from .vgflow2d import solve_vgflow2d_config


PERFORMANCE_FIELDS = ["case", "workload", "phase", "nx", "ny", "node_count", "element_count", "seconds", "notes"]


def run_vgflow2d_performance_benchmark(
    output_dir: str | Path,
    *,
    quick: bool = False,
    include_solve: bool = True,
) -> dict[str, Any]:
    """Run VGFlow 2D performance smoke/benchmark cases and write CSV/JSON."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in _benchmark_cases(quick):
        mesh = mesh_from_config(_mesh_config(case["nx"], case["ny"]))
        if include_solve:
            _record(records, case, "steady_solve", "cold", lambda c=case: _solve_case(c, out, "steady", "cold"))
            _record(records, case, "steady_solve", "warm", lambda c=case: _solve_case(c, out, "steady", "warm"))
            _record(records, case, "transient_multistep_post", "cold", lambda c=case: _solve_case(c, out, "transient", "cold"))
            _record(records, case, "transient_multistep_post", "warm", lambda c=case: _solve_case(c, out, "transient", "warm"))

        target_mesh, _manifest = upgrade_quad4_mesh_to_quad8(mesh)
        source_values = {nid: float(mesh.coords[mesh.node_index[nid], 0] + mesh.coords[mesh.node_index[nid], 1]) for nid in mesh.node_ids}
        plan_holder: dict[str, Any] = {}
        _record(records, case, "geofeas_projection_plan_build", "cold", lambda: plan_holder.setdefault("plan", build_scalar_projection_plan(mesh, target_mesh, locations="both")))
        plan = plan_holder["plan"]
        _record(records, case, "geofeas_projection_apply", "cold", lambda: apply_scalar_projection_plan(plan, source_values))
        scaled_values = {nid: value * 1.01 for nid, value in source_values.items()}
        _record(records, case, "geofeas_projection_apply", "warm", lambda: apply_scalar_projection_plan(plan, scaled_values))

        for record in records:
            if record["case"] == case["name"] and record["node_count"] == "":
                record["node_count"] = len(mesh.node_ids)
                record["element_count"] = len(mesh.elements)

    payload = {
        "schema": "geofem.vgflow2d.performance_benchmark.v1",
        "features": [
            "steady_and_transient_vgflow_timing",
            "projection_plan_build_and_reuse_timing",
            "cold_and_warm_phase_separation",
            "csv_json_performance_artifacts",
        ],
        "records": records,
        "diagnostics": {
            "case_count": len({row["case"] for row in records}),
            "record_count": len(records),
            "solve_included": bool(include_solve),
            "quick": bool(quick),
        },
    }
    json_path = out / "vgflow2d_performance.json"
    csv_path = out / "vgflow2d_performance.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERFORMANCE_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    payload["artifacts"] = {"json": str(json_path), "csv": str(csv_path)}
    return payload


def _benchmark_cases(quick: bool) -> Iterable[dict[str, Any]]:
    if quick:
        return [{"name": "quick", "nx": 1, "ny": 1}]
    return [
        {"name": "small", "nx": 4, "ny": 2},
        {"name": "medium", "nx": 12, "ny": 6},
        {"name": "large", "nx": 24, "ny": 12},
    ]


def _record(records: list[dict[str, Any]], case: dict[str, Any], workload: str, phase: str, fn: Callable[[], Any]) -> None:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    records.append(
        {
            "case": case["name"],
            "workload": workload,
            "phase": phase,
            "nx": int(case["nx"]),
            "ny": int(case["ny"]),
            "node_count": "",
            "element_count": "",
            "seconds": elapsed,
            "notes": "includes first-call compilation where applicable" if phase == "cold" else "warmed repeat",
        }
    )


def _mesh_config(nx: int, ny: int) -> dict[str, Any]:
    return {
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "nx": int(nx),
            "ny": int(ny),
            "element_type": "QUAD4",
            "material": "soil",
        }
    }


def _solve_case(case: dict[str, Any], output_dir: Path, mode: str, phase: str) -> None:
    cfg = _solve_config(int(case["nx"]), int(case["ny"]), mode)
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        solve_vgflow2d_config(cfg, Path(tmp) / f"{case['name']}_{mode}_{phase}")


def _solve_config(nx: int, ny: int, mode: str) -> dict[str, Any]:
    vgflow: dict[str, Any] = {
        "mode": mode,
        "problem_type": "vertical",
        "known_head_bcs": [{"set": "left", "head": 2.0}, {"set": "right", "head": 1.0}],
        "initial_water_level": 1.0,
    }
    if mode == "transient":
        vgflow["times"] = [0.0, 1.0, 2.0]
        vgflow["post"] = {"contour_level_count": 4, "flowline_seed_count": 4, "flowline_max_points": 8}
    return {
        "analysis": {"dimension": "2D", "type": "vgflow2d", "mode": mode},
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "nx": int(nx),
            "ny": int(ny),
            "element_type": "QUAD4",
            "material": "soil",
        },
        "materials": {
            "soil": {
                "model": "elastic",
                "E": 1000.0,
                "nu": 0.3,
                "seepage": {
                    "kx": 1.0e-5,
                    "ky": 1.0e-5,
                    "specific_storage": 1.0e-4,
                    "unsaturated": {"model": "van_genuchten", "alpha": 1.5, "n": 2.0, "theta_r": 0.1, "theta_s": 0.45},
                },
            }
        },
        "vgflow2d": vgflow,
    }


__all__ = ["PERFORMANCE_FIELDS", "run_vgflow2d_performance_benchmark"]
