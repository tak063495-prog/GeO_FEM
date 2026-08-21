"""Detached mesh worker helpers for GUI operations."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


class _TextValue:
    def __init__(self, value: Any) -> None:
        self._value = "" if value is None else str(value)

    def text(self) -> str:
        return self._value


class _CurrentTextValue:
    def __init__(self, value: Any) -> None:
        self._value = "" if value is None else str(value)

    def currentText(self) -> str:
        return self._value


def generate_auto_geometry_mesh_snapshot(
    window_cls: type[Any],
    cfg: Mapping[str, Any],
    *,
    requested_type: str,
    material: str,
    integration: str,
    boolean_expression: str = "",
    mesh_x0: str = "",
    mesh_x1: str = "",
    mesh_nx: str = "",
) -> dict[str, Any]:
    """Generate shape-based auto mesh on a detached MainWindow context."""

    from geofem_app.gui.model_check_worker import DetachedGuiContext

    context = DetachedGuiContext(window_cls, copy.deepcopy(dict(cfg)))
    context.mesh_type = _CurrentTextValue(requested_type)
    context.mesh_material = _TextValue(material)
    context.mesh_integration = _CurrentTextValue(integration)
    context.mesh_boolean_expression = _TextValue(boolean_expression)
    context.mesh_x0 = _TextValue(mesh_x0)
    context.mesh_x1 = _TextValue(mesh_x1)
    context.mesh_nx = _TextValue(mesh_nx)
    context._after_form_change = lambda _message: None
    window_cls.apply_auto_geometry_mesh(context)
    mesh = context.cfg.get("mesh", {})
    if not isinstance(mesh, Mapping):
        raise ValueError("auto mesh worker did not produce a mesh mapping")
    return {"mesh": copy.deepcopy(dict(mesh))}


def collect_mesh_quality_violations_snapshot(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    from geofem_app.fem2d import mesh_from_config
    from geofem_app.mesh_quality import evaluate_mesh_quality

    mesh = mesh_from_config(dict(cfg))
    return list(evaluate_mesh_quality(mesh, dict(cfg)).get("violations", []))


def compare_mesh_quality_improvements_snapshot(
    cfg: Mapping[str, Any],
    *,
    methods: Sequence[str],
    iterations: int,
    thresholds: tuple[float, float, float, float],
    selected_elements: Sequence[str],
) -> list[dict[str, Any]]:
    from geofem_app.fem2d import mesh_from_config
    from geofem_app.mesh_generation import improve_mesh_quality

    mesh_obj = mesh_from_config(dict(cfg))
    nodes, elements = _mesh_dict_from_fem_mesh(mesh_obj)
    min_area, min_angle, max_aspect, max_skew = thresholds
    rows: list[dict[str, Any]] = []
    selected = [str(element) for element in selected_elements]
    for method in methods:
        try:
            new_nodes, new_elements, report = improve_mesh_quality(
                nodes,
                elements,
                method=str(method),
                min_area=min_area,
                min_angle_deg=min_angle,
                max_aspect_ratio=max_aspect,
                max_skew=max_skew,
                iterations=max(1, int(iterations)),
                selected_elements=selected,
            )
        except Exception as exc:
            new_nodes, new_elements = nodes, elements
            report = {
                "method": str(method),
                "status": str(exc),
                "before": {},
                "after": {},
                "before_violation_count": "",
                "after_violation_count": "",
                "changed_nodes": 0,
                "changed_elements": 0,
                "score_delta": float("-inf"),
            }
        rows.append({"nodes": new_nodes, "elements": new_elements, "report": dict(report)})
    return rows


def _mesh_dict_from_fem_mesh(mesh: Any) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    nodes = {
        str(nid): [float(mesh.coords[index, 0]), float(mesh.coords[index, 1])]
        for index, nid in enumerate(mesh.node_ids)
    }
    elements: list[dict[str, Any]] = []
    for element in mesh.elements:
        item = {
            "id": str(element.id),
            "type": str(element.type),
            "nodes": [str(nid) for nid in element.nodes],
            "material": str(element.material),
            "integration": str(element.integration),
        }
        if not bool(getattr(element, "active", True)):
            item["active"] = False
        elements.append(item)
    return nodes, elements


__all__ = [
    "collect_mesh_quality_violations_snapshot",
    "compare_mesh_quality_improvements_snapshot",
    "generate_auto_geometry_mesh_snapshot",
]
