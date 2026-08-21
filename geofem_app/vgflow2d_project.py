"""Open project package I/O for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_types import Mesh2D
from .html_report_utils import report_css, table


PROJECT_SCHEMA = "geofem.vgflow2d.project_package.public_substitute.v1"


def write_vgflow_project_package(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, str]:
    paths = {
        "project_package_vg2": str(out / "vgflow_public_project.VG2"),
        "project_package_manifest": str(out / "vgflow_public_project_manifest.json"),
        "project_package_inventory": str(out / "vgflow_public_project_inventory.csv"),
        "project_package_html": str(out / "vgflow_public_project.html"),
    }
    package_path = Path(paths["project_package_vg2"])
    manifest = _project_manifest(package_path, mesh, materials, steps, problem_type, seepage, artifacts)
    members = _package_members(mesh, materials, steps, seepage, artifacts, manifest)
    _write_zip_package(package_path, members)
    Path(paths["project_package_manifest"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_inventory_csv(Path(paths["project_package_inventory"]), manifest)
    Path(paths["project_package_html"]).write_text(_html(manifest), encoding="utf-8")
    return paths


def read_vgflow_project_package(path: str | Path) -> dict[str, Any]:
    package_path = Path(path)
    with zipfile.ZipFile(package_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("schema") != PROJECT_SCHEMA:
            raise ValueError(f"unsupported VGFlow2D public project package schema: {manifest.get('schema')!r}")
        package = {
            "manifest": manifest,
            "model": json.loads(zf.read("model.json").decode("utf-8")),
            "seepage": json.loads(zf.read("seepage.json").decode("utf-8")),
        }
        for optional in ("steps.json", "artifacts.json"):
            if optional in zf.namelist():
                package[optional.removesuffix(".json")] = json.loads(zf.read(optional).decode("utf-8"))
    return package


def _project_manifest(
    package_path: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": PROJECT_SCHEMA,
        "profile": "Open VGFlow 2D substitute project package; not a proprietary commercial VG2 binary.",
        "filename": str(package_path),
        "extension": ".VG2",
        "commercial_vg2_binary_equivalence": False,
        "features": [
            "open_vg2_surrogate_zip_package",
            "roundtrip_readable_manifest_model_seepage",
            "artifact_inventory",
            "result_snapshot",
        ],
        "problem_type": problem_type,
        "analysis_mode": str(seepage.get("mode", seepage.get("analysis_mode", "steady"))),
        "node_count": len(mesh.node_ids),
        "element_count": len(mesh.elements),
        "material_count": len(materials),
        "step_count": len(steps),
        "members": [
            {"path": "manifest.json", "role": "package metadata"},
            {"path": "model.json", "role": "mesh and material snapshot"},
            {"path": "seepage.json", "role": "VGFlow2D public substitute seepage settings"},
            {"path": "steps.json", "role": "result step summaries"},
            {"path": "artifacts.json", "role": "output artifact inventory"},
        ],
        "artifact_count": len(artifacts),
    }


def _package_members(
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    payloads = {
        "manifest.json": manifest,
        "model.json": _model_snapshot(mesh, materials),
        "seepage.json": _jsonable(dict(seepage)),
        "steps.json": _step_snapshot(steps),
        "artifacts.json": {"artifacts": dict(sorted(artifacts.items()))},
    }
    return {name: json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8") for name, payload in payloads.items()}


def _write_zip_package(path: Path, members: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.date_time = (2026, 5, 23, 0, 0, 0)
            zf.writestr(info, members[name])


def _model_snapshot(mesh: Mesh2D, materials: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": nid, "x": float(mesh.coords[index, 0]), "y": float(mesh.coords[index, 1])}
            for index, nid in enumerate(mesh.node_ids)
        ],
        "elements": [
            {"id": element.id, "type": element.type, "material": element.material, "nodes": list(element.nodes)}
            for element in mesh.elements
        ],
        "node_sets": {key: list(value) for key, value in sorted(mesh.node_sets.items())},
        "element_sets": {key: list(value) for key, value in sorted(mesh.element_sets.items())},
        "materials": {name: _material_snapshot(material) for name, material in sorted(materials.items())},
    }


def _material_snapshot(material: Any) -> dict[str, Any]:
    rows = {}
    for key in (
        "name",
        "kx",
        "ky",
        "specific_storage",
        "theta_r",
        "theta_s",
        "alpha",
        "n",
        "angle_deg",
        "unsaturated_model",
        "table",
    ):
        rows[key] = _jsonable(getattr(material, key, None))
    return rows


def _step_snapshot(steps: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for step in steps:
        total_head = np.asarray(step.total_head, dtype=float)
        rows.append(
            {
                "index": int(step.index),
                "time": float(step.time),
                "iteration_count": int(getattr(step, "iteration_count", 0)),
                "residual_norm": float(getattr(step, "residual_norm", 0.0)),
                "active_seepage_nodes": int(getattr(step, "active_seepage_nodes", 0)),
                "total_head_min": float(np.min(total_head)) if total_head.size else 0.0,
                "total_head_max": float(np.max(total_head)) if total_head.size else 0.0,
                "total_head": [float(value) for value in total_head],
            }
        )
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(inner) for inner in value]
    return value


def _write_inventory_csv(path: Path, manifest: Mapping[str, Any]) -> None:
    fields = ["path", "role"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in manifest.get("members", []):
            writer.writerow({key: row.get(key, "") for key in fields})


def _html(manifest: Mapping[str, Any]) -> str:
    rows = [[row.get("path", ""), row.get("role", "")] for row in manifest.get("members", [])]
    summary = [
        ["schema", manifest.get("schema", "")],
        ["package", manifest.get("filename", "")],
        ["nodes", manifest.get("node_count", "")],
        ["elements", manifest.get("element_count", "")],
        ["steps", manifest.get("step_count", "")],
        ["commercial_vg2_binary_equivalence", manifest.get("commercial_vg2_binary_equivalence", "")],
    ]
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D Public Project Package</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D Public Project Package</h1>"
        "<p>商用VG2バイナリではなく、本ツール内で再読込できる公開代替VG2パッケージです。</p>"
        f"{table(['key', 'value'], summary)}"
        "<h2>Members</h2>"
        f"{table(['path', 'role'], rows)}"
        "</body></html>"
    )


__all__ = ["PROJECT_SCHEMA", "read_vgflow_project_package", "write_vgflow_project_package"]
