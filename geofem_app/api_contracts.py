"""Documented API contracts between GeoFEM responsibility boundaries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fem2d_types import Mesh2D, SolveResult2D, StageResult2D


@dataclass(frozen=True)
class ApiContract:
    name: str
    version: str
    owner: str
    producer: str
    consumers: tuple[str, ...]
    required_fields: tuple[str, ...]
    invariant: str


CONTRACTS: tuple[ApiContract, ...] = (
    ApiContract(
        name="input_config",
        version="geofem.input_config.v1",
        owner="configuration",
        producer="CLI/GUI/YAML loader",
        consumers=("input_diagnostics", "fem2d_mesh", "fem2d_solver", "reports"),
        required_fields=("analysis", "mesh", "materials"),
        invariant="2D input is a mapping with explicit analysis, mesh, and material sections.",
    ),
    ApiContract(
        name="mesh2d",
        version="geofem.mesh2d.v1",
        owner="mesh",
        producer="fem2d_mesh.mesh_from_config",
        consumers=("solver", "mesh_quality", "post", "reports"),
        required_fields=("node_ids", "coords", "elements", "node_sets", "element_sets"),
        invariant="coords is an Nx2 numeric array and every element references existing node ids.",
    ),
    ApiContract(
        name="material_table",
        version="geofem.material_table.v1",
        owner="materials",
        producer="fem2d_materials.materials_from_config",
        consumers=("solver", "reports", "material_models"),
        required_fields=("name", "E", "nu", "model"),
        invariant="Each material has positive stiffness and a valid Poisson ratio.",
    ),
    ApiContract(
        name="stage_result2d",
        version="geofem.stage_result2d.v1",
        owner="solver",
        producer="fem2d_solver",
        consumers=("post", "analysis_log", "performance_monitor", "reports"),
        required_fields=("name", "displacements", "reactions", "element_results", "solver_info"),
        invariant="Displacement and reaction vectors are finite arrays compatible with mesh degrees of freedom.",
    ),
    ApiContract(
        name="solve_result2d",
        version="geofem.solve_result2d.v1",
        owner="solver",
        producer="solve_plane_strain_config",
        consumers=("fem2d_io", "result_viewer", "standard_report", "benchmarks"),
        required_fields=("mesh", "materials", "stages", "output_dir", "warnings", "input_config"),
        invariant="A completed solve carries the mesh, material table, stage results, and output directory together.",
    ),
    ApiContract(
        name="analysis_artifact_bundle",
        version="geofem.analysis_artifact_bundle.v1",
        owner="output",
        producer="fem2d_io.write_run_summary",
        consumers=("GUI", "case_management", "external automation"),
        required_fields=("summary", "result_view_index", "standard_report", "performance"),
        invariant="Generated artifact paths are explicit strings and remain stable for GUI/result navigation.",
    ),
)


def api_contract_catalog() -> list[dict[str, Any]]:
    """Return all documented public responsibility-boundary contracts."""

    return [asdict(contract) for contract in CONTRACTS]


def contract_for(name: str) -> dict[str, Any]:
    for contract in CONTRACTS:
        if contract.name == name:
            return asdict(contract)
    raise KeyError(f"unknown GeoFEM API contract: {name}")


def validate_api_contract(name: str, payload: Any) -> dict[str, Any]:
    """Validate the small set of invariants that define a responsibility boundary."""

    issues: list[dict[str, str]] = []

    def add(path: str, message: str) -> None:
        issues.append({"path": path, "message": message})

    if name == "input_config":
        _validate_input_config(payload, add)
    elif name == "mesh2d":
        _validate_mesh2d(payload, add)
    elif name == "material_table":
        _validate_material_table(payload, add)
    elif name == "stage_result2d":
        _validate_stage_result2d(payload, add)
    elif name == "solve_result2d":
        _validate_solve_result2d(payload, add)
    elif name == "analysis_artifact_bundle":
        _validate_artifact_bundle(payload, add)
    else:
        add("contract", f"unknown contract '{name}'")

    return {
        "schema": "geofem.api_contract_validation.v1",
        "contract": name,
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def write_api_contract_docs(output_dir: str | Path) -> dict[str, str]:
    """Write JSON/CSV/Markdown contract documentation for developers."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = api_contract_catalog()
    json_path = out / "api_contracts.json"
    csv_path = out / "api_contracts.csv"
    md_path = out / "api_contracts.md"
    json_path.write_text(json.dumps({"schema": "geofem.api_contract_catalog.v1", "contracts": catalog}, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "version", "owner", "producer", "consumers", "required_fields", "invariant"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in catalog:
            writer.writerow(
                {
                    **row,
                    "consumers": ", ".join(row["consumers"]),
                    "required_fields": ", ".join(row["required_fields"]),
                }
            )
    md_path.write_text(_contract_markdown(catalog), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def _validate_input_config(payload: Any, add: Any) -> None:
    if not isinstance(payload, Mapping):
        add("input", "input config must be a mapping")
        return
    for key in ("analysis", "mesh", "materials"):
        if key not in payload:
            add(key, "required input section is missing")
    analysis = payload.get("analysis", {})
    if isinstance(analysis, Mapping) and str(analysis.get("dimension", "2D")).upper().replace(" ", "") != "2D":
        add("analysis.dimension", "only 2D is valid for this application boundary")


def _validate_mesh2d(payload: Any, add: Any) -> None:
    if not isinstance(payload, Mesh2D):
        add("mesh", "payload must be Mesh2D")
        return
    if payload.coords.ndim != 2 or payload.coords.shape[1] != 2:
        add("mesh.coords", "coords must be an Nx2 array")
    if len(payload.node_ids) != payload.coords.shape[0]:
        add("mesh.node_ids", "node_ids length must match coords rows")
    known = set(payload.node_ids)
    for element in payload.elements:
        missing = [node for node in element.nodes if node not in known]
        if missing:
            add(f"mesh.elements.{element.id}", "element references unknown nodes: " + ", ".join(missing))


def _validate_material_table(payload: Any, add: Any) -> None:
    if not isinstance(payload, Mapping):
        add("materials", "material table must be a mapping")
        return
    for name, mat in payload.items():
        for field in ("E", "nu"):
            value = getattr(mat, field, None) if not isinstance(mat, Mapping) else mat.get(field)
            if value is None:
                add(f"materials.{name}.{field}", "required material field is missing")
        E = getattr(mat, "E", None) if not isinstance(mat, Mapping) else mat.get("E")
        nu = getattr(mat, "nu", None) if not isinstance(mat, Mapping) else mat.get("nu")
        if _float_or_none(E) is not None and float(E) <= 0.0:
            add(f"materials.{name}.E", "E must be positive")
        if _float_or_none(nu) is not None and not 0.0 <= float(nu) < 0.5:
            add(f"materials.{name}.nu", "nu must satisfy 0 <= nu < 0.5")


def _validate_stage_result2d(payload: Any, add: Any) -> None:
    if not isinstance(payload, StageResult2D):
        add("stage", "payload must be StageResult2D")
        return
    if not payload.name:
        add("stage.name", "stage name is required")
    for field in ("displacements", "reactions"):
        value = getattr(payload, field)
        if not isinstance(value, np.ndarray) or not np.all(np.isfinite(value)):
            add(f"stage.{field}", f"{field} must be a finite numpy array")
    if not isinstance(payload.solver_info, dict):
        add("stage.solver_info", "solver_info must be a dictionary")


def _validate_solve_result2d(payload: Any, add: Any) -> None:
    if not isinstance(payload, SolveResult2D):
        add("result", "payload must be SolveResult2D")
        return
    _validate_mesh2d(payload.mesh, add)
    _validate_material_table(payload.materials, add)
    if not payload.stages:
        add("result.stages", "at least one stage result is required")
    for index, stage in enumerate(payload.stages):
        _validate_stage_result2d(stage, add)
        expected = len(payload.mesh.node_ids) * 2
        if stage.displacements.size < expected:
            add(f"result.stages[{index}].displacements", "displacement vector is shorter than mesh dof count")


def _validate_artifact_bundle(payload: Any, add: Any) -> None:
    if not isinstance(payload, Mapping):
        add("artifacts", "artifact bundle must be a mapping")
        return
    for key in ("summary", "result_view_index", "standard_report", "performance"):
        if key not in payload:
            add(key, "required artifact group is missing")


def _contract_markdown(catalog: list[dict[str, Any]]) -> str:
    lines = [
        "# GeoFEM API 契約",
        "",
        "この文書は `geofem_app.api_contracts` から生成できる責務境界の要約です。",
        "",
    ]
    for row in catalog:
        lines.extend(
            [
                f"## {row['name']}",
                f"- version: `{row['version']}`",
                f"- owner: {row['owner']}",
                f"- producer: {row['producer']}",
                f"- consumers: {', '.join(row['consumers'])}",
                f"- required_fields: {', '.join(row['required_fields'])}",
                f"- invariant: {row['invariant']}",
                "",
            ]
        )
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ApiContract",
    "api_contract_catalog",
    "contract_for",
    "validate_api_contract",
    "write_api_contract_docs",
]
