"""Detached Post export and report-audit helpers for GUI workers."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from geofem_app.gui.file_preview import preview_text_file


def save_post_image_snapshot(
    image: Any,
    path: str | Path,
    *,
    role: str = "post_image",
    image_format: str = "PNG",
) -> dict[str, Any]:
    """Save a prepared QImage without touching GUI widgets."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".png" and image_format.upper() == "PNG":
        out = out.with_suffix(".png")
    if image is None or image.isNull():
        raise ValueError("empty post image")
    if not image.save(str(out), image_format.upper()):
        raise OSError(f"failed to save post image: {out}")
    return {
        "operation": "save_image",
        "role": role,
        "path": str(out),
        "width": int(image.width()),
        "height": int(image.height()),
        "bytes": out.stat().st_size if out.exists() else 0,
    }


def compare_post_image_snapshot(
    current_image: Any,
    baseline: str | Path,
    *,
    threshold: float = 0.02,
    delta_threshold: int = 18,
) -> dict[str, Any]:
    """Run pixel comparison away from the GUI thread."""

    from geofem_app.post_image_diff import compare_images

    result = dict(compare_images(current_image, baseline, threshold=threshold, delta_threshold=delta_threshold))
    result.update({"operation": "compare_image", "baseline": str(Path(baseline))})
    return result


def export_scene_pdf_snapshot(
    path: str | Path,
    *,
    current_image: Any | None = None,
    layout_specs: Sequence[Mapping[str, Any]] | None = None,
    snapshot_paths: Sequence[str | Path] | None = None,
    page_title: str = "GeoFEM 2D Post",
    resolution: int = 160,
) -> dict[str, Any]:
    """Write a PDF drawing from pre-rendered images and layout metadata."""

    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPageSize, QPainter, QPdfWriter, QPen

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".pdf":
        out = out.with_suffix(".pdf")

    writer = QPdfWriter(str(out))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setResolution(int(resolution))
    painter = QPainter(writer)
    page_count = 0
    used_snapshots = 0
    specs = [_normalize_layout_spec(spec) for spec in (layout_specs or [])]
    try:
        if specs:
            _draw_layout_pdf_page(painter, specs, page_title=page_title)
            page_count = 1
        else:
            if current_image is None or current_image.isNull():
                raise ValueError("empty post image")
            painter.drawImage(painter.viewport(), current_image)
            page_count = 1
            for snapshot in snapshot_paths or []:
                snapshot_image = QImage(str(snapshot))
                if snapshot_image.isNull():
                    continue
                writer.newPage()
                painter.drawImage(painter.viewport(), snapshot_image)
                page_count += 1
                used_snapshots += 1
    finally:
        painter.end()

    return {
        "operation": "scene_pdf",
        "path": str(out),
        "page_count": page_count,
        "layout_count": len(specs),
        "snapshot_count": used_snapshots,
        "bytes": out.stat().st_size if out.exists() else 0,
        "features": ["worker_pdf_writer", "pre_rendered_scene_image", "layout_manifest_input"],
    }


def build_selected_report_snapshot(
    *,
    results_dir: str | Path,
    stage_dir: str | Path | None = None,
    include_summary: bool = True,
    include_tables: bool = True,
) -> dict[str, Any]:
    """Build the GUI-selected HTML report from frozen file paths."""

    results = Path(results_dir)
    commercial_report = results / "calculation_report.html"
    if commercial_report.exists():
        return {"operation": "build_report", "path": str(commercial_report), "existing": True}
    summary_path = results / "summary.json"
    stage = Path(stage_dir) if stage_dir else None
    parts = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>GeoFEM 2D Report</title>',
        "<style>body{font-family:Meiryo,sans-serif;line-height:1.55}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:5px}th{background:#f3f4f6}pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #ddd;padding:10px}</style><body>",
        "<h1>GeoFEM 2D Report</h1>",
    ]
    if include_summary and summary_path.exists():
        summary = html.escape(preview_text_file(summary_path).text)
        parts.extend(["<h2>Analysis summary</h2><pre>", summary, "</pre>"])
    if include_tables and stage is not None:
        parts.append("<h2>Result table links</h2><ul>")
        for name in ["displacements.csv", "reactions.csv", "element_stress.csv", "interface_state.csv", "pore_pressure.csv", "riks_path.csv"]:
            candidate = stage / name
            if candidate.exists():
                parts.append(f"<li>{html.escape(name)}: {html.escape(str(candidate))}</li>")
        parts.append("</ul>")
    run_log = results / "run.log"
    if run_log.exists():
        parts.extend(["<h2>Log</h2><pre>", html.escape(preview_text_file(run_log).text), "</pre>"])
    if stage is not None and (stage / "report.html").exists():
        parts.append(f"<p>Stage report: {html.escape(str(stage / 'report.html'))}</p>")
    parts.append("</body></html>")
    out = results / "gui_report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return {"operation": "build_report", "path": str(out), "existing": False, "bytes": out.stat().st_size}


def copy_report_pdf_snapshot(source: str | Path, destination: str | Path, *, manifest: str | Path | None = None) -> dict[str, Any]:
    """Copy an existing calculation-report PDF in a worker."""

    src = Path(source)
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != out.resolve():
        shutil.copyfile(src, out)
    return {
        "operation": "copy_report_pdf",
        "source": str(src),
        "path": str(out),
        "manifest": str(manifest or ""),
        "bytes": out.stat().st_size if out.exists() else 0,
    }


def audit_post_report_snapshot(
    *,
    result_dir: str | Path,
    output_dir: str | Path,
    baseline_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the GeoFEAS-style Post/report audit in a worker."""

    from geofem_app.geofeas_verification import audit_post_report_package

    out = Path(output_dir)
    summary = dict(audit_post_report_package(result_dir, baseline_dir=baseline_dir, output_dir=out))
    return {
        "operation": "report_audit",
        "summary": summary,
        "paths": {
            "json": str(out / "geofeas_post_report_audit.json"),
            "csv": str(out / "geofeas_post_report_audit.csv"),
            "html": str(out / "geofeas_post_report_audit.html"),
        },
    }


def _normalize_layout_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    rect_raw = spec.get("rect", (0.05, 0.05, 0.9, 0.4))
    try:
        x, y, w, h = rect_raw  # type: ignore[misc]
        rect = (
            max(0.0, min(1.0, float(x))),
            max(0.0, min(1.0, float(y))),
            max(0.05, min(1.0, float(w))),
            max(0.05, min(1.0, float(h))),
        )
    except Exception:
        rect = (0.05, 0.05, 0.9, 0.4)
    return {
        "path": str(spec.get("path", "")),
        "title": str(spec.get("title", "")),
        "scale": str(spec.get("scale", "")),
        "rect": rect,
    }


def _draw_layout_pdf_page(painter: Any, specs: Sequence[Mapping[str, Any]], *, page_title: str) -> None:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPen

    viewport = painter.viewport()
    painter.fillRect(viewport, QColor("#ffffff"))
    painter.setPen(QPen(QColor("#111111")))
    painter.drawRect(viewport.adjusted(8, 8, -8, -8))
    painter.drawText(24, 32, page_title)
    for spec in specs:
        x, y, w, h = spec["rect"]
        target = QRectF(
            viewport.left() + viewport.width() * x,
            viewport.top() + viewport.height() * y,
            viewport.width() * w,
            viewport.height() * h,
        )
        image = QImage(str(spec["path"]))
        if image.isNull():
            continue
        painter.drawImage(target, image)
        painter.drawRect(target)
        painter.drawText(int(target.left()), int(target.bottom()) + 16, f"{spec['title']}  {spec['scale']}")


__all__ = [
    "audit_post_report_snapshot",
    "build_selected_report_snapshot",
    "compare_post_image_snapshot",
    "copy_report_pdf_snapshot",
    "export_scene_pdf_snapshot",
    "save_post_image_snapshot",
]
