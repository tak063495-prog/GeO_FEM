"""Result-table routing rules for the GUI.

The main window should decide how to present a selected result, while this
module owns the stable mapping from a result kind to its CSV artifact.
"""

from __future__ import annotations

from pathlib import Path


STAGE_RESULT_FILES: dict[str, str] = {
    "displacements": "displacements.csv",
    "displacement_contour": "displacements.csv",
    "displacement_vectors": "displacements.csv",
    "element_stress": "element_stress.csv",
    "plastic": "element_stress.csv",
    "safety_factor": "element_stress.csv",
    "interface_state": "interface_state.csv",
    "structural_state": "structural_state.csv",
    "structural_section_forces": "structural_section_forces.csv",
    "pore_pressure": "pore_pressure.csv",
    "reactions": "reactions.csv",
    "riks_path": "riks_path.csv",
}

ROOT_RESULT_FILES: dict[str, str] = {
    "mesh_quality": "mesh_quality.csv",
    "analysis_log": "analysis_log.csv",
    "performance": "performance_summary.csv",
    "large_model_operations": "large_model_operations.csv",
    "node_search_index": "large_model_node_index.csv",
    "element_search_index": "large_model_element_index.csv",
    "standard_report": "standard_report_sections.csv",
}


def is_root_result_kind(kind: str) -> bool:
    return kind in ROOT_RESULT_FILES


def result_table_path(kind: str, *, stage_dir: Path, summary_path: Path | None) -> Path:
    """Resolve the CSV path for a GUI result-table kind."""

    if kind in ROOT_RESULT_FILES:
        if summary_path is None:
            raise ValueError("summary_path is required for root-level result tables")
        return summary_path.parent / ROOT_RESULT_FILES[kind]
    if kind in STAGE_RESULT_FILES:
        return stage_dir / STAGE_RESULT_FILES[kind]
    raise KeyError(f"unknown result table kind: {kind}")


def known_result_table_kinds() -> list[str]:
    return sorted(set(STAGE_RESULT_FILES) | set(ROOT_RESULT_FILES))


__all__ = ["ROOT_RESULT_FILES", "STAGE_RESULT_FILES", "is_root_result_kind", "known_result_table_kinds", "result_table_path"]
