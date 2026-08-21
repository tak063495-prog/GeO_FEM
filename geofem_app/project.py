"""Project file helpers for the GeoFEM GUI/CLI layer."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_EXTENSION = ".gfemproj"
PROJECT_DIRS = ("input", "mesh", "runs", "results", "reports", "logs")


@dataclass
class GeoFEMProject:
    name: str
    dimension: str = "2D"
    unit_system: str = "m-kN"
    analysis_type: str = "static_plane_strain"
    project_file: str | None = None
    input_file: str | None = None
    latest_run: str | None = None
    recent_runs: list[str] = field(default_factory=list)
    run_records: list[dict[str, Any]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["format"] = "GeoFEM project"
        data["format_version"] = 1
        return data


def ensure_project_dirs(root: str | Path) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_DIRS:
        (root_path / name).mkdir(parents=True, exist_ok=True)


def load_project(path: str | Path) -> GeoFEMProject:
    project_path = Path(path)
    data = json.loads(project_path.read_text(encoding="utf-8"))
    data.pop("format", None)
    data.pop("format_version", None)
    project = GeoFEMProject(**data)
    project.project_file = str(project_path)
    return project


def save_project(project: GeoFEMProject, path: str | Path | None = None) -> Path:
    if path is None:
        if not project.project_file:
            raise ValueError("project path is required")
        path = project.project_file
    project_path = Path(path)
    if project_path.suffix.lower() != PROJECT_EXTENSION:
        project_path = project_path.with_suffix(PROJECT_EXTENSION)
    ensure_project_dirs(project_path.parent)
    if project_path.exists():
        backup = project_path.with_suffix(project_path.suffix + ".bak")
        backup.write_text(project_path.read_text(encoding="utf-8"), encoding="utf-8")
    project.project_file = str(project_path)
    project_path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return project_path


def new_default_project(root: str | Path, *, name: str = "GeoFEM Project", dimension: str = "2D") -> GeoFEMProject:
    root_path = Path(root)
    ensure_project_dirs(root_path)
    dimension_norm = dimension.upper()
    if dimension_norm != "2D":
        raise ValueError("3D projects have been removed; only 2D projects are supported")
    project = GeoFEMProject(name=name, dimension="2D", analysis_type="static_plane_strain")
    project.project_file = str(root_path / f"{_safe_filename(name)}{PROJECT_EXTENSION}")
    return project


def project_root(project: GeoFEMProject) -> Path:
    if project.project_file:
        return Path(project.project_file).resolve().parent
    return Path.cwd()


def update_after_run(project: GeoFEMProject, run_dir: str | Path) -> None:
    run_str = str(Path(run_dir))
    project.latest_run = run_str
    project.recent_runs = [run_str] + [p for p in project.recent_runs if p != run_str]
    project.recent_runs = project.recent_runs[:10]


def add_run_record(project: GeoFEMProject, record: Mapping[str, Any]) -> None:
    project.run_records = [dict(record)] + [row for row in project.run_records if row.get("output_dir") != record.get("output_dir")]
    project.run_records = project.run_records[:50]


def project_from_mapping(data: Mapping[str, Any]) -> GeoFEMProject:
    dimension = str(data.get("dimension", "2D")).upper()
    if dimension != "2D":
        raise ValueError("3D projects have been removed; only 2D projects are supported")
    return GeoFEMProject(
        name=str(data.get("name", "GeoFEM Project")),
        dimension="2D",
        unit_system=str(data.get("unit_system", "m-kN")),
        analysis_type=str(data.get("analysis_type", "static_plane_strain")),
        model=dict(data.get("model", {})) if isinstance(data.get("model", {}), Mapping) else {},
    )


def _safe_filename(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return text or "project"
