"""Post image generation and pixel-diff helpers for GUI regression checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _qimage_from(value: str | Path | Any) -> Any:
    from PySide6.QtGui import QImage

    if isinstance(value, QImage):
        return value
    return QImage(str(value))


def create_sample_post_image(path: str | Path, *, width: int = 900, height: int = 640) -> Path:
    """Create a deterministic Post-style image used by CI smoke diffs."""

    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#ffffff"))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.fillRect(0, 0, width, height, QColor("#ffffff"))
    painter.setPen(QPen(QColor("#111827"), 2))
    painter.drawRect(24, 24, width - 48, height - 48)

    painter.fillRect(44, 45, 260, 18, QColor("#111827"))
    painter.fillRect(44, 72, 180, 8, QColor("#6c757d"))

    mesh_left = 70
    mesh_top = 120
    mesh_w = width - 230
    mesh_h = height - 190
    painter.setPen(QPen(QColor("#495057"), 1))
    for i in range(8):
        x = mesh_left + int(mesh_w * i / 7)
        painter.drawLine(x, mesh_top, x + 22, mesh_top + mesh_h)
    for j in range(6):
        y = mesh_top + int(mesh_h * j / 5)
        painter.drawLine(mesh_left, y, mesh_left + mesh_w + 22, y + 20)

    colors = ["#2b8cbe", "#4eb3d3", "#7bccc4", "#a8ddb5", "#ccebc5", "#ffffcc", "#fed976", "#feb24c", "#fd8d3c", "#f03b20"]
    for idx, color in enumerate(colors):
        y = mesh_top + int(mesh_h * idx / len(colors))
        painter.setPen(QPen(QColor(color), 3))
        painter.drawArc(QRectF(mesh_left + 30 + idx * 8, y, mesh_w - idx * 16, 92), 0, 16 * 180)

    painter.setPen(QPen(QColor("#6c757d"), 2))
    painter.drawLine(mesh_left, mesh_top + mesh_h + 28, mesh_left + mesh_w, mesh_top + mesh_h + 28)
    painter.drawLine(mesh_left + mesh_w + 8, mesh_top + mesh_h + 24, mesh_left + mesh_w + 44, mesh_top + mesh_h + 24)

    legend_x = width - 130
    legend_y = 132
    legend_h = 300
    band_h = legend_h / len(colors)
    for idx, color in enumerate(reversed(colors)):
        painter.fillRect(int(legend_x), int(legend_y + idx * band_h), 30, int(band_h + 1), QBrush(QColor(color)))
    painter.setPen(QPen(QColor("#111827"), 1))
    painter.drawRect(legend_x, legend_y, 30, legend_h)
    painter.fillRect(legend_x + 42, legend_y + 4, 38, 8, QColor("#111827"))
    painter.fillRect(legend_x + 42, legend_y + legend_h - 8, 30, 8, QColor("#111827"))

    painter.end()
    image.save(str(out))
    return out


def compare_images(
    current: str | Path | Any,
    baseline: str | Path | Any,
    *,
    threshold: float = 0.02,
    delta_threshold: int = 18,
) -> dict[str, Any]:
    """Compare two images and return a compact regression summary."""

    image_a = _qimage_from(current)
    image_b = _qimage_from(baseline)
    if image_a.isNull() or image_b.isNull():
        return {"ok": False, "reason": "missing image"}

    width = min(image_a.width(), image_b.width())
    height = min(image_a.height(), image_b.height())
    total = max(width * height, 1)
    diff = 0
    accum = 0.0
    for y in range(height):
        for x in range(width):
            ca = image_a.pixelColor(x, y)
            cb = image_b.pixelColor(x, y)
            delta = abs(ca.red() - cb.red()) + abs(ca.green() - cb.green()) + abs(ca.blue() - cb.blue()) + abs(ca.alpha() - cb.alpha())
            accum += delta
            if delta > delta_threshold:
                diff += 1

    ratio = diff / total
    same_size = image_a.width() == image_b.width() and image_a.height() == image_b.height()
    return {
        "ok": same_size and ratio <= threshold,
        "diff_ratio": ratio,
        "mean_delta": accum / total,
        "changed_pixels": diff,
        "samples": total,
        "threshold": threshold,
        "delta_threshold": delta_threshold,
        "current_size": [image_a.width(), image_a.height()],
        "baseline_size": [image_b.width(), image_b.height()],
    }
