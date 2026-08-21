"""Display quality policy helpers for large GUI models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayQualityPolicy:
    mode: str
    node_count: int
    element_count: int
    draw_node_labels: bool
    draw_element_labels: bool
    draw_element_boundaries: bool
    draw_contour_labels: bool
    max_vectors: int
    contour_level_count: int
    contour_curve_segments: int
    reduced: bool
    batch_mesh_items: bool
    batch_node_items: bool
    pick_nearest_mesh: bool


def resolve_display_quality_policy(
    *,
    mode: str,
    node_count: int,
    element_count: int,
    detail_limit: int,
    vector_limit: int,
    requested_node_labels: bool,
    requested_element_labels: bool,
    requested_element_boundaries: bool,
    requested_contour_labels: bool,
    requested_contour_levels: int,
    requested_curve_segments: int,
) -> DisplayQualityPolicy:
    """Resolve user display settings into a concrete drawing policy."""

    normalized = str(mode or "auto").lower()
    detail = max(50, int(detail_limit))
    vectors = max(25, int(vector_limit))
    levels = max(2, min(30, int(requested_contour_levels)))
    curve_segments = max(2, min(16, int(requested_curve_segments)))
    node_total = max(0, int(node_count))
    element_total = max(0, int(element_count))
    model_size = max(node_total, element_total)
    reduced = normalized == "fast" or (normalized == "auto" and model_size > detail)

    if normalized == "full":
        return DisplayQualityPolicy(
            mode="full",
            node_count=node_total,
            element_count=element_total,
            draw_node_labels=requested_node_labels,
            draw_element_labels=requested_element_labels,
            draw_element_boundaries=requested_element_boundaries,
            draw_contour_labels=requested_contour_labels,
            max_vectors=max(vectors, node_total),
            contour_level_count=levels,
            contour_curve_segments=curve_segments,
            reduced=False,
            batch_mesh_items=False,
            batch_node_items=False,
            pick_nearest_mesh=False,
        )

    if reduced:
        aggressive = normalized == "fast"
        boundary_limit = detail if aggressive else detail * 2
        return DisplayQualityPolicy(
            mode="fast" if aggressive else "auto",
            node_count=node_total,
            element_count=element_total,
            draw_node_labels=False,
            draw_element_labels=False,
            draw_element_boundaries=requested_element_boundaries and element_total <= boundary_limit,
            draw_contour_labels=False,
            max_vectors=min(vectors, max(25, detail // (3 if aggressive else 2))),
            contour_level_count=min(levels, 5 if aggressive else 8),
            contour_curve_segments=min(curve_segments, 2 if aggressive else 3),
            reduced=True,
            batch_mesh_items=True,
            batch_node_items=True,
            pick_nearest_mesh=True,
        )

    return DisplayQualityPolicy(
        mode="auto",
        node_count=node_total,
        element_count=element_total,
        draw_node_labels=requested_node_labels,
        draw_element_labels=requested_element_labels,
        draw_element_boundaries=requested_element_boundaries,
        draw_contour_labels=requested_contour_labels,
        max_vectors=vectors,
        contour_level_count=levels,
        contour_curve_segments=curve_segments,
        reduced=False,
        batch_mesh_items=False,
        batch_node_items=False,
        pick_nearest_mesh=False,
    )


__all__ = ["DisplayQualityPolicy", "resolve_display_quality_policy"]
