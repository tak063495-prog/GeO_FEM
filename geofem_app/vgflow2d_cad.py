"""CAD and raster intake diagnostics for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cad_import import CadImportError, cad_document_line_tuples, parse_cad_file_document
from .fem2d_utils import _ensure_list
from .html_report_utils import report_css


LineRow = dict[str, Any]


def write_vgflow_cad_import_outputs(out: Path, seepage: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "cad_import_diagnostics_json": str(out / "vgflow_cad_import_diagnostics.json"),
        "cad_import_diagnostics_csv": str(out / "vgflow_cad_import_diagnostics.csv"),
        "cad_import_model_lines_csv": str(out / "vgflow_cad_import_model_lines.csv"),
        "cad_import_html": str(out / "vgflow_cad_import.html"),
    }
    diagnostics, lines = collect_vgflow_cad_import_diagnostics(seepage)
    payload = {
        "schema": "geofem.vgflow2d.cad_import.public_substitute.v1",
        "features": [
            "shared_geofem_cad_import_engine",
            "cad_exchange_attribute_summary",
            "dxf_mm_to_m_scale_correction",
            "raster_image_calibration",
            "raster_traced_polyline_to_model_lines",
            "raster_auto_dark_line_extraction",
        ],
        "shared_engine": shared_vgflow_cad_import_engine(),
        "diagnostics": diagnostics,
        "model_line_count": len(lines),
    }
    Path(paths["cad_import_diagnostics_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_diagnostics_csv(Path(paths["cad_import_diagnostics_csv"]), diagnostics)
    _write_lines_csv(Path(paths["cad_import_model_lines_csv"]), lines)
    Path(paths["cad_import_html"]).write_text(_html_report(payload, lines), encoding="utf-8")
    return paths


def collect_vgflow_cad_import_diagnostics(seepage: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[LineRow]]:
    cfg = _cad_cfg(seepage)
    diagnostics: list[dict[str, Any]] = []
    model_lines: list[LineRow] = []
    for index, spec in enumerate(_cad_file_specs(cfg)):
        rows, lines = _diagnose_cad_file(index, spec)
        diagnostics.extend(rows)
        model_lines.extend(lines)
    for index, spec in enumerate(_raster_specs(cfg)):
        rows, lines = _diagnose_raster(index, spec)
        diagnostics.extend(rows)
        model_lines.extend(lines)
    if not diagnostics:
        diagnostics.append(_row("cad_import", "pass", message="No VGFlow2D CAD or raster import inputs were provided."))
    diagnostics.insert(
        0,
        _row(
            "shared_geofem_cad_import_engine",
            "pass",
            message="VGFlow2D CAD intake uses the same shared GeoFEM/GeoFEAS CAD import engine.",
            details=shared_vgflow_cad_import_engine(),
        ),
    )
    return diagnostics, model_lines


def shared_vgflow_cad_import_engine() -> dict[str, Any]:
    return {
        "module": "geofem_app.cad_import",
        "document_entrypoint": "parse_cad_file_document",
        "line_extractor": "cad_document_line_tuples",
        "shared_with": ["GeoFEAS public substitute", "GeoFEM GUI external data tab"],
        "open_formats": ["DXF", "SXF", "SFC", "P21", "GF1", "CSV/TXT line lists"],
        "dwg_strategy": "external converter through GEOFEM_DWG_CONVERTER, then shared DXF/SXF import",
        "commercial_vgflow_screen_equivalence": False,
    }


def _cad_cfg(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("cad_import", "vgflow_cad_import", "cad", "drawing_import"):
        value = seepage.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _cad_file_specs(cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = cfg.get("files", cfg.get("input_files", cfg.get("cad_files", [])))
    specs: list[Mapping[str, Any]] = []
    if cfg.get("file") or cfg.get("path"):
        specs.append(cfg)
    for item in _ensure_list(raw):
        if isinstance(item, Mapping):
            specs.append(item)
        elif item not in (None, ""):
            specs.append({"path": str(item)})
    return specs


def _raster_specs(cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = cfg.get("raster_images", cfg.get("image_import", cfg.get("images", [])))
    specs: list[Mapping[str, Any]] = []
    if _has_raster_fields(cfg):
        specs.append(cfg)
    for item in _ensure_list(raw):
        if isinstance(item, Mapping):
            specs.append(item)
        elif item not in (None, ""):
            specs.append({"source": str(item)})
    return specs


def _has_raster_fields(cfg: Mapping[str, Any]) -> bool:
    return any(key in cfg for key in ("source_image", "image", "raster", "calibration", "traced_polylines", "detected_polylines", "auto_extract", "auto_detect", "extract_strata"))


def _diagnose_cad_file(index: int, spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[LineRow]]:
    rows: list[dict[str, Any]] = []
    lines: list[LineRow] = []
    source = Path(str(spec.get("path", spec.get("file", spec.get("source", ""))))).expanduser()
    source_unit = _unit_name(spec.get("source_unit", spec.get("unit", spec.get("drawing_unit", ""))))
    target_unit = _unit_name(spec.get("target_unit", spec.get("model_unit", "m"))) or "m"
    suggested_scale = _unit_scale(source_unit, target_unit)
    explicit_scale = _float_or_none(spec.get("scale", spec.get("scale_factor")))
    applied_scale = explicit_scale if explicit_scale is not None else suggested_scale
    if not str(source):
        rows.append(_row("cad_file", "warning", source="", message="CAD import spec has no path."))
        return rows, lines
    if not source.exists():
        rows.append(_row("cad_file_exists", "error", source=str(source), message="CAD file was not found."))
        return rows, lines
    try:
        doc = parse_cad_file_document(source)
    except (CadImportError, OSError, ValueError) as exc:
        rows.append(_row("cad_file_parse", "error", source=str(source), message=str(exc)))
        return rows, lines

    raw_lines = cad_document_line_tuples(doc)
    layer_count = len(_list(doc.get("layers", [])))
    annotation_count = len(_list(doc.get("annotations", [])))
    dimension_count = len(_list(doc.get("dimensions", [])))
    rows.append(
        _row(
            "cad_attribute_summary",
            "pass",
            source=str(source),
            value=len(raw_lines),
            message="CAD exchange file was parsed for VGFlow2D model creation diagnostics.",
            details={
                "format": doc.get("format", source.suffix.lower().lstrip(".")),
                "layer_count": layer_count,
                "curve_count": len(_list(doc.get("curves", []))),
                "region_count": len(_list(doc.get("regions", []))),
                "annotation_count": annotation_count,
                "dimension_count": dimension_count,
            },
        )
    )
    if source.suffix.lower() == ".dxf" and source_unit == "mm" and target_unit == "m":
        status = "pass" if abs(applied_scale - 0.001) <= 1.0e-15 else "warning"
        rows.append(
            _row(
                "dxf_mm_to_m_scale_correction",
                status,
                source=str(source),
                value=applied_scale,
                limit=0.001,
                message="DXF millimeter drawing units are converted to the VGFlow2D meter model unit." if status == "pass" else "DXF appears to use millimeters; review the scale factor before model creation.",
            )
        )
    elif source_unit and target_unit and abs(suggested_scale - 1.0) > 1.0e-15:
        rows.append(
            _row(
                "cad_unit_scale_correction",
                "pass",
                source=str(source),
                value=applied_scale,
                limit=suggested_scale,
                message=f"CAD coordinates are scaled from {source_unit} to {target_unit}.",
            )
        )
    for line_index, (x1, y1, x2, y2) in enumerate(raw_lines, start=1):
        lines.append(
            {
                "source_type": "cad",
                "source": str(source),
                "name": spec.get("name", source.stem),
                "line_index": line_index,
                "x1": x1 * applied_scale,
                "y1": y1 * applied_scale,
                "x2": x2 * applied_scale,
                "y2": y2 * applied_scale,
                "scale": applied_scale,
                "source_unit": source_unit,
                "target_unit": target_unit,
            }
        )
    return rows, lines


def _diagnose_raster(index: int, spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[LineRow]]:
    rows: list[dict[str, Any]] = []
    lines: list[LineRow] = []
    source = str(spec.get("source", spec.get("source_image", spec.get("image", spec.get("raster", f"raster-{index + 1}")))))
    transform = _raster_transform(spec)
    if transform is None:
        rows.append(
            _row(
                "raster_image_calibration",
                "warning",
                source=source,
                message="Raster image import requires two pixel/world calibration pairs before traced lines can be used as model geometry.",
            )
        )
        return rows, lines
    rows.append(
        _row(
            "raster_image_calibration",
            "pass",
            source=source,
            value=transform["scale"],
            message="Raster image coordinates were calibrated to VGFlow2D model coordinates.",
            details={"pixel_origin": transform["p1"], "world_origin": transform["q1"]},
        )
    )
    polylines = spec.get("traced_polylines", spec.get("detected_polylines", spec.get("polylines", [])))
    line_index = 1
    for poly_index, polyline in enumerate(_ensure_list(polylines), start=1):
        name, points = _polyline_points(polyline)
        if len(points) < 2:
            continue
        line_index = _append_raster_lines(lines, "raster", source, name or f"polyline-{poly_index}", points, transform, spec, line_index)
    rows.append(
        _row(
            "raster_traced_polyline_to_model_lines",
            "pass" if lines else "warning",
            source=source,
            value=len(lines),
            message="Raster traced polylines were converted to VGFlow2D model line candidates." if lines else "Raster calibration exists, but no traced polylines were provided.",
        )
    )
    auto_requested = bool(spec.get("auto_extract", spec.get("auto_detect", spec.get("extract_strata", False))))
    if auto_requested:
        auto = _auto_extract_raster_polylines(source, spec)
        auto_line_count_before = len(lines)
        for poly_index, points in enumerate(auto["polylines"], start=1):
            line_index = _append_raster_lines(lines, "raster_auto", source, f"auto_stratum_{poly_index}", points, transform, spec, line_index)
        extracted_count = len(lines) - auto_line_count_before
        rows.append(
            _row(
                "raster_auto_dark_line_extraction",
                "pass" if extracted_count else auto["status"],
                source=source,
                value=extracted_count,
                message=auto["message"] if not extracted_count else "Raster dark pixels were automatically vectorized into VGFlow2D model line candidates.",
                details=auto["details"],
            )
        )
    return rows, lines


def _append_raster_lines(
    lines: list[LineRow],
    source_type: str,
    source: str,
    name: str,
    points: Sequence[tuple[float, float]],
    transform: Mapping[str, Any],
    spec: Mapping[str, Any],
    line_index: int,
) -> int:
    world_points = [_apply_raster_transform(transform, point) for point in points]
    for start, end in zip(world_points[:-1], world_points[1:]):
        lines.append(
            {
                "source_type": source_type,
                "source": source,
                "name": name,
                "line_index": line_index,
                "x1": start[0],
                "y1": start[1],
                "x2": end[0],
                "y2": end[1],
                "scale": transform["scale"],
                "source_unit": "pixel",
                "target_unit": spec.get("target_unit", "m"),
            }
        )
        line_index += 1
    return line_index


def _raster_transform(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    pairs = _ensure_list(spec.get("calibration", spec.get("calibration_points", [])))
    if len(pairs) < 2:
        return None
    p1, q1 = _calibration_pair(pairs[0])
    p2, q2 = _calibration_pair(pairs[1])
    if p1 is None or p2 is None or q1 is None or q2 is None:
        return None
    dp = complex(p2[0] - p1[0], p2[1] - p1[1])
    dq = complex(q2[0] - q1[0], q2[1] - q1[1])
    if abs(dp) <= 1.0e-30:
        return None
    ratio = dq / dp
    return {"p1": p1, "q1": q1, "ratio": ratio, "scale": abs(ratio)}


def _calibration_pair(raw: Any) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if isinstance(raw, Mapping):
        pixel = _xy(raw.get("pixel", raw.get("image", raw.get("screen"))))
        world = _xy(raw.get("world", raw.get("model", raw.get("coord"))))
        return pixel, world
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        return _xy(raw[:2]), _xy(raw[2:4])
    return None, None


def _apply_raster_transform(transform: Mapping[str, Any], point: tuple[float, float]) -> tuple[float, float]:
    p1 = transform["p1"]
    q1 = transform["q1"]
    mapped = complex(q1[0], q1[1]) + complex(point[0] - p1[0], point[1] - p1[1]) * transform["ratio"]
    return float(mapped.real), float(mapped.imag)


def _polyline_points(raw: Any) -> tuple[str, list[tuple[float, float]]]:
    if isinstance(raw, Mapping):
        name = str(raw.get("name", raw.get("id", "")))
        points = [_xy(point) for point in _ensure_list(raw.get("points", raw.get("vertices", [])))]
    else:
        name = ""
        points = [_xy(point) for point in _ensure_list(raw)]
    return name, [point for point in points if point is not None]


def _auto_extract_raster_polylines(source: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    image_path = Path(source).expanduser()
    if not image_path.exists():
        return {
            "status": "warning",
            "message": "Raster auto extraction was requested, but the image file was not found.",
            "polylines": [],
            "details": {"source": source},
        }
    image = _load_grayscale_image(image_path)
    if image is None:
        return {
            "status": "warning",
            "message": "Raster auto extraction supports PGM/PPM directly and common images when Pillow is installed.",
            "polylines": [],
            "details": {"source": source, "format": image_path.suffix.lower()},
        }
    threshold = int(spec.get("threshold", spec.get("dark_threshold", 128)) or 128)
    min_pixels = max(1, int(spec.get("min_dark_pixels_per_column", spec.get("min_column_pixels", 1)) or 1))
    max_points = max(2, int(spec.get("max_auto_points", 200) or 200))
    polyline = _column_dark_centerline(image, threshold, min_pixels)
    simplified = _simplify_polyline(polyline, max_points)
    return {
        "status": "pass" if len(simplified) >= 2 else "warning",
        "message": "Raster dark pixels were automatically vectorized into VGFlow2D model line candidates." if len(simplified) >= 2 else "Raster image was readable, but no continuous dark stratum line was detected.",
        "polylines": [simplified] if len(simplified) >= 2 else [],
        "details": {
            "source": source,
            "width": len(image[0]) if image else 0,
            "height": len(image),
            "threshold": threshold,
            "min_dark_pixels_per_column": min_pixels,
            "point_count": len(simplified),
        },
    }


def _load_grayscale_image(path: Path) -> list[list[int]] | None:
    if path.suffix.lower() in {".pgm", ".ppm", ".pbm"}:
        return _load_netpbm_grayscale(path)
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            values = list(gray.getdata())
            return [[int(values[y * width + x]) for x in range(width)] for y in range(height)]
    except Exception:
        return None


def _load_netpbm_grayscale(path: Path) -> list[list[int]] | None:
    data = path.read_bytes()
    tokens = _netpbm_tokens(data)
    if len(tokens) < 4:
        return None
    magic = tokens[0].decode("ascii", errors="ignore")
    if magic not in {"P1", "P2", "P3"}:
        return None
    width = int(tokens[1])
    height = int(tokens[2])
    if width <= 0 or height <= 0:
        return None
    if magic == "P1":
        values = [0 if int(token) else 255 for token in tokens[3 : 3 + width * height]]
        max_value = 255
    elif magic == "P2":
        max_value = max(1, int(tokens[3]))
        values = [int(round(int(token) / max_value * 255.0)) for token in tokens[4 : 4 + width * height]]
    else:
        max_value = max(1, int(tokens[3]))
        raw = [int(token) for token in tokens[4 : 4 + width * height * 3]]
        values = []
        for index in range(0, len(raw), 3):
            r, g, b = raw[index : index + 3]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            values.append(int(round(gray / max_value * 255.0)))
    if len(values) < width * height:
        return None
    return [values[y * width : (y + 1) * width] for y in range(height)]


def _netpbm_tokens(data: bytes) -> list[bytes]:
    tokens: list[bytes] = []
    token = bytearray()
    in_comment = False
    for byte in data:
        if in_comment:
            if byte in (10, 13):
                in_comment = False
            continue
        if byte == 35:
            if token:
                tokens.append(bytes(token))
                token.clear()
            in_comment = True
            continue
        if byte <= 32:
            if token:
                tokens.append(bytes(token))
                token.clear()
        else:
            token.append(byte)
    if token:
        tokens.append(bytes(token))
    return tokens


def _column_dark_centerline(image: Sequence[Sequence[int]], threshold: int, min_pixels: int) -> list[tuple[float, float]]:
    if not image or not image[0]:
        return []
    width = len(image[0])
    points: list[tuple[float, float]] = []
    for x in range(width):
        ys = [y for y, row in enumerate(image) if x < len(row) and int(row[x]) <= threshold]
        if len(ys) >= min_pixels:
            points.append((float(x), float(sum(ys) / len(ys))))
    return _largest_contiguous_run(points)


def _largest_contiguous_run(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    runs: list[list[tuple[float, float]]] = []
    current = [points[0]]
    for point in points[1:]:
        if point[0] <= current[-1][0] + 1.0:
            current.append(point)
        else:
            runs.append(current)
            current = [point]
    runs.append(current)
    return max(runs, key=len)


def _simplify_polyline(points: Sequence[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return list(points)
    stride = max(1, int(round(len(points) / max_points)))
    simplified = list(points[::stride])
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified


def _unit_name(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _unit_scale(source_unit: str, target_unit: str) -> float:
    source = _unit_to_m(source_unit)
    target = _unit_to_m(target_unit)
    if source is None or target is None or target == 0.0:
        return 1.0
    return source / target


def _unit_to_m(unit: str) -> float | None:
    aliases = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "metre": 1.0,
        "mm": 0.001,
        "millimeter": 0.001,
        "millimeters": 0.001,
        "millimetre": 0.001,
        "cm": 0.01,
        "centimeter": 0.01,
        "centimeters": 0.01,
        "km": 1000.0,
    }
    return aliases.get(unit)


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        x = value.get("x", value.get("X"))
        y = value.get("y", value.get("Y"))
        if x is None or y is None:
            return None
        return float(x), float(y)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _row(
    check: str,
    status: str,
    *,
    source: str = "",
    value: Any = "",
    limit: Any = "",
    message: str = "",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check": check,
        "status": status,
        "source": source,
        "value": value,
        "limit": limit,
        "message": message,
        "details": dict(details or {}),
    }


def _write_diagnostics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["check", "status", "source", "value", "limit", "message", "details"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in fields}
            payload["details"] = json.dumps(payload.get("details", {}), ensure_ascii=False, default=str)
            writer.writerow(payload)


def _write_lines_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["source_type", "source", "name", "line_index", "x1", "y1", "x2", "y2", "scale", "source_unit", "target_unit"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _html_report(payload: Mapping[str, Any], lines: Sequence[Mapping[str, Any]]) -> str:
    diag_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('check', '')))}</td>"
        f"<td>{html.escape(str(row.get('status', '')))}</td>"
        f"<td>{html.escape(str(row.get('source', '')))}</td>"
        f"<td>{html.escape(str(row.get('message', '')))}</td>"
        "</tr>"
        for row in payload.get("diagnostics", [])
    )
    line_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('source_type', '')))}</td>"
        f"<td>{html.escape(str(row.get('name', '')))}</td>"
        f"<td>{float(row.get('x1', 0.0)):.8g}</td><td>{float(row.get('y1', 0.0)):.8g}</td>"
        f"<td>{float(row.get('x2', 0.0)):.8g}</td><td>{float(row.get('y2', 0.0)):.8g}</td>"
        "</tr>"
        for row in lines[:200]
    )
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D CAD Import Diagnostics</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D CAD Import Diagnostics</h1>"
        "<h2>Diagnostics</h2>"
        f"<table><thead><tr><th>check</th><th>status</th><th>source</th><th>message</th></tr></thead><tbody>{diag_rows}</tbody></table>"
        "<h2>Model Lines</h2>"
        f"<table><thead><tr><th>source</th><th>name</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead><tbody>{line_rows}</tbody></table>"
        "</body></html>"
    )
