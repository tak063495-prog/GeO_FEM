"""Lightweight AVI animation writer for VGFlow 2D public substitute Post output."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_types import Mesh2D


def write_vgflow_animation_avi(
    avi_path: Path,
    manifest_path: Path,
    mesh: Mesh2D,
    steps: Sequence[Any],
    problem_type: str,
    post: Mapping[str, Any],
) -> dict[str, Any]:
    width = max(96, int(post.get("video_width", post.get("animation_width", 320)) or 320))
    height = max(72, int(post.get("video_height", post.get("animation_height", 240)) or 240))
    fps = max(1, int(post.get("video_fps", post.get("animation_fps", 4)) or 4))
    frames = [_render_head_frame(mesh, step, problem_type, width, height, index, len(steps)) for index, step in enumerate(steps)]
    _write_uncompressed_avi(avi_path, frames, fps)
    manifest = {
        "schema": "geofem.vgflow2d.animation_avi.public_substitute.v1",
        "profile": "Direct lightweight RIFF/AVI DIB animation generated from VGFlow 2D public substitute total-head results.",
        "format": "AVI",
        "codec": "DIB",
        "commercial_renderer_equivalence": False,
        "frame_count": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
        "problem_type": problem_type,
        "artifact": str(avi_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def _write_uncompressed_avi(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("VGFlow2D AVI export requires at least one frame")
    normalized = [np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8) for frame in frames]
    height, width, _ = normalized[0].shape
    if any(frame.shape != normalized[0].shape for frame in normalized):
        raise ValueError("all VGFlow2D AVI frames must have the same shape")
    row_stride = ((width * 3 + 3) // 4) * 4
    image_size = row_stride * height
    frame_payloads = [_rgb_to_dib_frame(frame, row_stride) for frame in normalized]
    frame_chunks = [_chunk(b"00db", payload) for payload in frame_payloads]
    avih = struct.pack(
        "<14I",
        int(1_000_000 / fps),
        image_size * fps,
        0,
        0x10,
        len(frame_payloads),
        0,
        1,
        image_size,
        width,
        height,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIiiii",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        len(frame_payloads),
        image_size,
        0xFFFFFFFF,
        0,
        0,
        0,
        width,
        height,
    )
    strf = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        height,
        1,
        24,
        0,
        image_size,
        0,
        0,
        0,
        0,
    )
    hdrl = _list_chunk(b"hdrl", _chunk(b"avih", avih) + _list_chunk(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))
    movi_payload = b"".join(frame_chunks)
    movi = _list_chunk(b"movi", movi_payload)
    idx_entries = []
    offset = 4
    for payload, chunk in zip(frame_payloads, frame_chunks):
        idx_entries.append(struct.pack("<4sIII", b"00db", 0x10, offset, len(payload)))
        offset += len(chunk)
    idx1 = _chunk(b"idx1", b"".join(idx_entries))
    riff_payload = hdrl + movi + idx1
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff_payload) + 4) + b"AVI " + riff_payload)


def _rgb_to_dib_frame(frame: np.ndarray, row_stride: int) -> bytes:
    height, width, _ = frame.shape
    raw = np.zeros((height, row_stride), dtype=np.uint8)
    bgr = frame[::-1, :, ::-1].reshape(height, width * 3)
    raw[:, : width * 3] = bgr
    return raw.tobytes()


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")


def _list_chunk(list_type: bytes, payload: bytes) -> bytes:
    data = list_type + payload
    return b"LIST" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def _render_head_frame(mesh: Mesh2D, step: Any, problem_type: str, width: int, height: int, frame_index: int, frame_count: int) -> np.ndarray:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_panel(frame)
    coords = mesh.coords
    x_min, y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
    x_max, y_max = np.max(coords[:, 0]), np.max(coords[:, 1])
    pad = max(16, min(width, height) // 12)
    usable_w = max(1.0, float(width - 2 * pad))
    usable_h = max(1.0, float(height - 2 * pad - 18))
    sx = usable_w / max(float(x_max - x_min), np.finfo(float).eps)
    sy = usable_h / max(float(y_max - y_min), np.finfo(float).eps)
    scale = min(sx, sy)
    x_offset = pad + 0.5 * (usable_w - float(x_max - x_min) * scale)
    y_offset = pad + 18 + 0.5 * (usable_h - float(y_max - y_min) * scale)
    head = np.asarray(step.total_head, dtype=float)
    head_min = float(np.min(head))
    head_max = float(np.max(head))
    values = (head - head_min) / max(head_max - head_min, np.finfo(float).eps)
    pixels = [_project(coords[i, 0], coords[i, 1], x_min, y_min, x_offset, y_offset, scale, height, pad) for i in range(len(mesh.node_ids))]
    _draw_elements(frame, mesh, pixels)
    for (x, y), value in zip(pixels, values):
        _draw_square(frame, x, y, 3, _color_map(float(value)))
    _draw_progress(frame, frame_index, max(frame_count, 1))
    _draw_time_ticks(frame, frame_index, float(getattr(step, "time", frame_index)))
    return frame


def _project(x: float, y: float, x_min: float, y_min: float, x_offset: float, y_offset: float, scale: float, height: int, pad: int) -> tuple[int, int]:
    px = int(round(x_offset + (float(x) - x_min) * scale))
    py = int(round(height - pad - (float(y) - y_min) * scale - (y_offset - pad - 18)))
    return px, py


def _draw_panel(frame: np.ndarray) -> None:
    frame[:16, :, :] = np.array([241, 245, 249], dtype=np.uint8)
    frame[16:18, :, :] = np.array([203, 213, 225], dtype=np.uint8)


def _draw_elements(frame: np.ndarray, mesh: Mesh2D, pixels: Sequence[tuple[int, int]]) -> None:
    node_to_pixel = {nid: pixels[i] for i, nid in enumerate(mesh.node_ids)}
    for element in mesh.elements:
        nodes = list(element.nodes[:4] if element.type.upper().startswith("QUAD") else element.nodes[:3])
        for a, b in zip(nodes, [*nodes[1:], nodes[0]]):
            _draw_line(frame, *node_to_pixel[a], *node_to_pixel[b], (71, 85, 105))


def _draw_progress(frame: np.ndarray, frame_index: int, frame_count: int) -> None:
    height, width, _ = frame.shape
    x0, x1 = 10, width - 10
    y0, y1 = 6, 11
    frame[y0:y1, x0:x1, :] = np.array([226, 232, 240], dtype=np.uint8)
    filled = x0 + int((x1 - x0) * (frame_index + 1) / frame_count)
    frame[y0:y1, x0:filled, :] = np.array([37, 99, 235], dtype=np.uint8)
    frame[height - 8 : height - 4, 8 : width - 8, :] = np.array([226, 232, 240], dtype=np.uint8)


def _draw_time_ticks(frame: np.ndarray, frame_index: int, time_value: float) -> None:
    signature = int(abs(time_value) * 1000.0) + frame_index * 17
    for bit in range(16):
        if signature & (1 << bit):
            x = 8 + bit * 5
            frame[20:25, x : x + 3, :] = np.array([15, 23, 42], dtype=np.uint8)


def _color_map(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        t = value * 2.0
        return (_byte(59 + t * 41), _byte(130 + t * 86), _byte(246 - t * 98))
    t = (value - 0.5) * 2.0
    return (_byte(100 + t * 155), _byte(216 - t * 178), _byte(148 - t * 110))


def _byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _draw_square(frame: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    height, width, _ = frame.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    frame[y0:y1, x0:x1, :] = np.asarray(color, dtype=np.uint8)


def _draw_line(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < frame.shape[1] and 0 <= y0 < frame.shape[0]:
            frame[y0, x0, :] = np.asarray(color, dtype=np.uint8)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


__all__ = ["write_vgflow_animation_avi"]
