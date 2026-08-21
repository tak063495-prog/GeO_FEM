"""Lightweight CAD/exchange line import helpers for the 2D GUI."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from .cad_dwg_converter import (
    DEFAULT_DWG_CONVERTER_CANDIDATES,
    discover_dwg_converter as _discover_dwg_converter_core,
    dwg_converter_candidates as _dwg_converter_candidates_core,
    dwg_converter_command as _dwg_converter_command_core,
    dwg_converter_requirement_message as _dwg_converter_requirement_message_core,
)
from .cad_gf1_payload import (
    best_effort_decode as _best_effort_decode_core,
    decoded_text_candidates as _decoded_text_candidates_core,
    decode_with_encoding as _decode_with_encoding_core,
    gf1_binary_payload_marker as _gf1_binary_payload_marker_core,
    gf1_payload_blobs as _gf1_payload_blobs_core,
    gf1_payload_text_candidates as _gf1_payload_text_candidates_core,
    gf1_text_candidates as _gf1_text_candidates_core,
    json_payloads as _json_payloads_core,
    looks_binary as _looks_binary_core,
    looks_like_gf1_text as _looks_like_gf1_text_core,
    zlib_payloads as _zlib_payloads_core,
)

Line2D = tuple[float, float, float, float]
CadDocument = dict[str, Any]


class CadImportError(ValueError):
    """Raised when a requested exchange format cannot be imported directly."""


def parse_cad_lines(text: str, suffix: str) -> list[Line2D]:
    ext = suffix.lower().strip()
    if ext == ".dwg":
        raise CadImportError("DWG binary import requires an external converter; use parse_cad_file with GEOFEM_DWG_CONVERTER or convert DWG to DXF/SXF first")
    if ext == ".dxf":
        return parse_dxf_lines(text)
    if ext in {".sxf", ".sfc", ".p21", ".stp", ".step"}:
        return parse_sxf_lines(text)
    if ext in {".gf1", ".json", ".yaml", ".yml"}:
        parsed = parse_gf1_lines(text)
        return parsed if parsed else parse_plain_lines(text)
    return parse_plain_lines(text)


def parse_cad_document(text: str, suffix: str) -> CadDocument:
    ext = suffix.lower().strip()
    if ext == ".dwg":
        raise CadImportError("DWG binary import requires an external converter; use parse_cad_file_document with GEOFEM_DWG_CONVERTER")
    if ext == ".dxf":
        return parse_dxf_document(text)
    if ext in {".sxf", ".sfc", ".p21", ".stp", ".step"}:
        return parse_sxf_document(text)
    if ext in {".gf1", ".json", ".yaml", ".yml"}:
        doc = parse_gf1_document(text)
        return doc if _document_has_content(doc) else document_from_lines(parse_plain_lines(text), fmt="plain")
    return document_from_lines(parse_plain_lines(text), fmt="plain")


def parse_cad_file(path: str | Path, *, converter: str | Sequence[str] | None = None, output_suffix: str | None = None) -> list[Line2D]:
    source = Path(path)
    ext = source.suffix.lower().strip()
    if ext == ".dwg":
        return parse_dwg_file(source, converter=converter, output_suffix=output_suffix)
    if ext == ".gf1":
        data = source.read_bytes()
        parsed = parse_gf1_lines_bytes(data)
        return parsed if parsed else parse_plain_lines(_best_effort_decode(data))
    text = source.read_text(encoding="utf-8", errors="ignore")
    return parse_cad_lines(text, ext)


def parse_cad_file_document(path: str | Path, *, converter: str | Sequence[str] | None = None, output_suffix: str | None = None) -> CadDocument:
    source = Path(path)
    ext = source.suffix.lower().strip()
    if ext == ".dwg":
        return parse_dwg_file_document(source, converter=converter, output_suffix=output_suffix)
    if ext == ".gf1":
        data = source.read_bytes()
        doc = parse_gf1_document_bytes(data)
        if not _document_has_content(doc):
            doc = document_from_lines(parse_plain_lines(_best_effort_decode(data)), fmt="plain")
        doc["source"] = str(source)
        return doc
    text = source.read_text(encoding="utf-8", errors="ignore")
    doc = parse_cad_document(text, ext)
    doc["source"] = str(source)
    return doc


def parse_dwg_file(path: str | Path, *, converter: str | Sequence[str] | None = None, output_suffix: str | None = None) -> list[Line2D]:
    return cad_document_line_tuples(parse_dwg_file_document(path, converter=converter, output_suffix=output_suffix))


def parse_dwg_file_document(path: str | Path, *, converter: str | Sequence[str] | None = None, output_suffix: str | None = None) -> CadDocument:
    source = Path(path)
    if not source.exists():
        raise CadImportError(f"DWG file not found: {source}")
    suffix = (output_suffix or os.environ.get("GEOFEM_DWG_OUTPUT_SUFFIX") or ".dxf").lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    converter_cmd = converter if converter is not None else discover_dwg_converter()
    if not converter_cmd:
        raise CadImportError(dwg_converter_requirement_message())
    with tempfile.TemporaryDirectory(prefix="geofem_dwg_") as tmp:
        output = Path(tmp) / f"converted{suffix}"
        cmd = _dwg_converter_command(converter_cmd, source, output)
        try:
            completed = subprocess.run(
                cmd,
                cwd=source.parent,
                capture_output=True,
                text=True,
                timeout=float(os.environ.get("GEOFEM_DWG_CONVERTER_TIMEOUT", "60")),
                shell=isinstance(cmd, str),
                check=False,
            )
        except OSError as exc:
            raise CadImportError(f"DWG converter could not be started: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise CadImportError("DWG converter timed out") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise CadImportError(f"DWG converter failed with code {completed.returncode}: {detail}")
        if output.exists():
            converted = output.read_text(encoding="utf-8", errors="ignore")
            doc = parse_cad_document(converted, output.suffix)
            doc["source"] = str(source)
            doc["converted_suffix"] = output.suffix
            return doc
        if completed.stdout.strip():
            doc = parse_cad_document(completed.stdout, suffix)
            doc["source"] = str(source)
            doc["converted_suffix"] = suffix
            return doc
        raise CadImportError(f"DWG converter did not produce {output.name}")


def _dwg_converter_command(converter: str | Sequence[str], source: Path, output: Path) -> str | list[str]:
    return _dwg_converter_command_core(converter, source, output)


def discover_dwg_converter() -> str | Sequence[str] | None:
    return _discover_dwg_converter_core()


def _dwg_converter_candidates() -> list[str]:
    return _dwg_converter_candidates_core()


def dwg_converter_requirement_message() -> str:
    return _dwg_converter_requirement_message_core()


def parse_plain_lines(text: str) -> list[Line2D]:
    out: list[Line2D] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part for part in stripped.replace(";", ",").replace("\t", ",").split(",") if part.strip()]
        if len(parts) < 4:
            parts = [part for part in stripped.split() if part.strip()]
        if len(parts) < 4:
            continue
        try:
            out.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    return out


def parse_dxf_lines(text: str) -> list[Line2D]:
    tokens = [line.strip() for line in text.splitlines()]
    out: list[Line2D] = []
    i = 0
    entity_names = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ENDSEC", "EOF"}
    while i + 1 < len(tokens):
        code = tokens[i]
        value = tokens[i + 1].upper()
        if code == "0" and value == "LINE":
            i += 2
            vals: dict[str, float] = {}
            while i + 1 < len(tokens) and not (tokens[i] == "0" and tokens[i + 1].upper() in entity_names):
                if tokens[i] in {"10", "20", "11", "21"}:
                    _assign_float(vals, tokens[i], tokens[i + 1])
                i += 2
            if all(key in vals for key in ("10", "20", "11", "21")):
                out.append((vals["10"], vals["20"], vals["11"], vals["21"]))
            continue
        if code == "0" and value == "LWPOLYLINE":
            i += 2
            points: list[tuple[float, float]] = []
            closed = False
            pending_x: float | None = None
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] == "70":
                    closed = _closed_flag(tokens[i + 1])
                elif tokens[i] == "10":
                    pending_x = _float_or_none(tokens[i + 1])
                elif tokens[i] == "20" and pending_x is not None:
                    y = _float_or_none(tokens[i + 1])
                    if y is not None:
                        points.append((pending_x, y))
                    pending_x = None
                i += 2
            out.extend(segments_from_points(points, closed=closed))
            continue
        if code == "0" and value == "POLYLINE":
            i += 2
            points = []
            closed = False
            while i + 1 < len(tokens):
                if tokens[i] == "70":
                    closed = _closed_flag(tokens[i + 1])
                if tokens[i] == "0" and tokens[i + 1].upper() == "VERTEX":
                    i += 2
                    vals: dict[str, float] = {}
                    while i + 1 < len(tokens) and tokens[i] != "0":
                        if tokens[i] in {"10", "20"}:
                            _assign_float(vals, tokens[i], tokens[i + 1])
                        i += 2
                    if "10" in vals and "20" in vals:
                        points.append((vals["10"], vals["20"]))
                    continue
                if tokens[i] == "0" and tokens[i + 1].upper() == "SEQEND":
                    i += 2
                    break
                if tokens[i] == "0" and tokens[i + 1].upper() in entity_names:
                    break
                i += 2
            out.extend(segments_from_points(points, closed=closed))
            continue
        if code == "0" and value in {"ARC", "CIRCLE"}:
            entity = value
            i += 2
            vals: dict[str, float] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] in {"10", "20", "40", "50", "51"}:
                    _assign_float(vals, tokens[i], tokens[i + 1])
                i += 2
            if all(key in vals for key in ("10", "20", "40")):
                out.extend(arc_segments(vals["10"], vals["20"], vals["40"], vals.get("50", 0.0), vals.get("51", 360.0), full_circle=(entity == "CIRCLE")))
            continue
        i += 2
    return out


def parse_dxf_document(text: str) -> CadDocument:
    tokens = [line.strip() for line in text.splitlines()]
    doc = empty_cad_document("dxf")
    entity_names = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "TEXT", "MTEXT", "DIMENSION", "HATCH", "SOLID", "ENDSEC", "EOF"}
    i = 0
    while i + 1 < len(tokens):
        code = tokens[i]
        value = tokens[i + 1].upper()
        if code == "0" and value == "LTYPE":
            i += 2
            attrs: dict[str, Any] = {}
            pattern: list[float] = []
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] == "2":
                    attrs["name"] = tokens[i + 1]
                elif tokens[i] == "3":
                    attrs["description"] = tokens[i + 1]
                elif tokens[i] == "40":
                    _assign_float(attrs, "pattern_length", tokens[i + 1])
                elif tokens[i] == "49":
                    value49 = _float_or_none(tokens[i + 1])
                    if value49 is not None:
                        pattern.append(value49)
                i += 2
            name = str(attrs.pop("name", "")).strip()
            if name:
                add_document_linetype(doc, name, pattern=pattern, **attrs)
            continue
        if code == "0" and value == "LAYER":
            i += 2
            attrs: dict[str, Any] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] == "2":
                    attrs["name"] = tokens[i + 1]
                elif tokens[i] == "6":
                    attrs["linetype"] = tokens[i + 1]
                elif tokens[i] == "62":
                    attrs["color"] = dxf_color_name(tokens[i + 1])
                    attrs["aci_color"] = _int_or_none(tokens[i + 1])
                    attrs["visible"] = (_int_or_none(tokens[i + 1]) or 0) >= 0
                elif tokens[i] == "370":
                    attrs["lineweight"] = _int_or_none(tokens[i + 1])
                i += 2
            name = str(attrs.pop("name", "")).strip()
            if name:
                add_document_layer(doc, name, **attrs)
            continue
        if code == "0" and value == "DIMSTYLE":
            i += 2
            attrs: dict[str, Any] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] == "2":
                    attrs["name"] = tokens[i + 1]
                elif tokens[i] == "3":
                    attrs["postfix"] = tokens[i + 1]
                elif tokens[i] == "40":
                    _assign_float(attrs, "scale", tokens[i + 1])
                elif tokens[i] == "41":
                    _assign_float(attrs, "arrow_size", tokens[i + 1])
                elif tokens[i] == "140":
                    _assign_float(attrs, "text_height", tokens[i + 1])
                elif tokens[i] in {"176", "177", "178"}:
                    attrs["color"] = dxf_color_name(tokens[i + 1])
                i += 2
            name = str(attrs.pop("name", "")).strip()
            if name:
                add_document_dimension_style(doc, name, **attrs)
            continue
        if code == "0" and value == "LINE":
            i += 2
            vals: dict[str, Any] = {}
            while i + 1 < len(tokens) and not (tokens[i] == "0" and tokens[i + 1].upper() in entity_names):
                if tokens[i] in {"10", "20", "11", "21"}:
                    _assign_float(vals, tokens[i], tokens[i + 1])
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            if all(key in vals for key in ("10", "20", "11", "21")):
                add_document_line(doc, (vals["10"], vals["20"], vals["11"], vals["21"]), **_style_kwargs_from_mapping(vals))
            continue
        if code == "0" and value == "LWPOLYLINE":
            i += 2
            vals: dict[str, Any] = {}
            points: list[tuple[float, float]] = []
            closed = False
            pending_x: float | None = None
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] == "70":
                    closed = _closed_flag(tokens[i + 1])
                elif tokens[i] == "10":
                    pending_x = _float_or_none(tokens[i + 1])
                elif tokens[i] == "20" and pending_x is not None:
                    y = _float_or_none(tokens[i + 1])
                    if y is not None:
                        points.append((pending_x, y))
                    pending_x = None
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            for segment in segments_from_points(points, closed=closed):
                add_document_line(doc, segment, **_style_kwargs_from_mapping(vals))
            continue
        if code == "0" and value == "POLYLINE":
            i += 2
            vals: dict[str, Any] = {}
            points = []
            closed = False
            while i + 1 < len(tokens):
                if tokens[i] == "70":
                    closed = _closed_flag(tokens[i + 1])
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                if tokens[i] == "0" and tokens[i + 1].upper() == "VERTEX":
                    i += 2
                    vertex: dict[str, float] = {}
                    while i + 1 < len(tokens) and tokens[i] != "0":
                        if tokens[i] in {"10", "20"}:
                            _assign_float(vertex, tokens[i], tokens[i + 1])
                        i += 2
                    if "10" in vertex and "20" in vertex:
                        points.append((vertex["10"], vertex["20"]))
                    continue
                if tokens[i] == "0" and tokens[i + 1].upper() == "SEQEND":
                    i += 2
                    break
                if tokens[i] == "0" and tokens[i + 1].upper() in entity_names:
                    break
                i += 2
            for segment in segments_from_points(points, closed=closed):
                add_document_line(doc, segment, **_style_kwargs_from_mapping(vals))
            continue
        if code == "0" and value in {"ARC", "CIRCLE"}:
            entity = value
            i += 2
            vals: dict[str, Any] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] in {"10", "20", "40", "50", "51"}:
                    _assign_float(vals, tokens[i], tokens[i + 1])
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            if all(key in vals for key in ("10", "20", "40")):
                for segment in arc_segments(vals["10"], vals["20"], vals["40"], vals.get("50", 0.0), vals.get("51", 360.0), full_circle=(entity == "CIRCLE")):
                    add_document_line(doc, segment, **_style_kwargs_from_mapping(vals))
            continue
        if code == "0" and value in {"TEXT", "MTEXT"}:
            i += 2
            vals: dict[str, Any] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] in {"10", "20"}:
                    _assign_float(vals, tokens[i], tokens[i + 1])
                elif tokens[i] in {"1", "3"}:
                    vals["text"] = f"{vals.get('text', '')}{tokens[i + 1]}"
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            if "10" in vals and "20" in vals and str(vals.get("text", "")).strip():
                add_document_annotation(doc, (vals["10"], vals["20"]), str(vals["text"]), **_style_kwargs_from_mapping(vals))
            continue
        if code == "0" and value == "DIMENSION":
            i += 2
            vals: dict[str, Any] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] in {"10", "20", "11", "21", "13", "23", "14", "24"}:
                    _assign_float(vals, tokens[i], tokens[i + 1])
                elif tokens[i] == "1":
                    vals["text"] = tokens[i + 1]
                elif tokens[i] == "3":
                    vals["dimension_style"] = tokens[i + 1]
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            start = _dxf_dimension_point(vals, "13", "23") or _dxf_dimension_point(vals, "10", "20")
            end = _dxf_dimension_point(vals, "14", "24") or _dxf_dimension_point(vals, "11", "21")
            if start is not None and end is not None:
                add_document_dimension(doc, start, end, text=vals.get("text"), **_dimension_kwargs_from_mapping(vals))
            continue
        if code == "0" and value == "HATCH":
            i += 2
            vals: dict[str, Any] = {}
            points: list[tuple[float, float]] = []
            rings: list[list[tuple[float, float]]] = []
            pending_x: float | None = None
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] == "2":
                    vals["pattern"] = tokens[i + 1]
                elif tokens[i] == "92":
                    if len(points) >= 3:
                        rings.append(points)
                    points = []
                elif tokens[i] == "10":
                    pending_x = _float_or_none(tokens[i + 1])
                elif tokens[i] == "20" and pending_x is not None:
                    y = _float_or_none(tokens[i + 1])
                    if y is not None:
                        points.append((pending_x, y))
                    pending_x = None
                elif tokens[i] == "70":
                    vals["solid"] = bool(_int_or_none(tokens[i + 1]) or 0)
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            if len(points) >= 3:
                rings.append(points)
            if rings:
                add_document_hatch(doc, rings=rings, **_hatch_kwargs_from_mapping(vals))
            continue
        if code == "0" and value == "SOLID":
            i += 2
            vals: dict[str, Any] = {"solid": True}
            coords: dict[str, float] = {}
            while i + 1 < len(tokens) and tokens[i] != "0":
                if tokens[i] in {"10", "20", "11", "21", "12", "22", "13", "23"}:
                    _assign_float(coords, tokens[i], tokens[i + 1])
                _assign_dxf_entity_attr(vals, tokens[i], tokens[i + 1])
                i += 2
            points = _solid_points(coords)
            if len(points) >= 3:
                add_document_hatch(doc, points, **_hatch_kwargs_from_mapping(vals))
            continue
        i += 2
    if _document_has_content(doc):
        return doc
    return document_from_lines(parse_dxf_lines(text), fmt="dxf")


def parse_sxf_lines(text: str) -> list[Line2D]:
    return cad_document_line_tuples(parse_sxf_document(text))


def parse_sxf_document(text: str) -> CadDocument:
    records = split_sxf_step_records(text)
    points: dict[str, tuple[float, float]] = {}
    layer_names: dict[str, str] = {}
    linetype_names: dict[str, str] = {}
    doc = empty_cad_document("sxf")
    doc["step_header"] = [record for record in records if _record_id(record) is None and record.upper().startswith(("ISO-", "HEADER", "FILE_", "DATA", "ENDSEC"))]
    for record in records:
        upper = record.upper()
        ident = _record_id(record)
        entity = _sxf_entity_item(record)
        if entity is not None:
            doc["entities"].append(entity)
        if ident and ("LINETYPE" in upper or "LINE_TYPE" in upper or "LINESTYLE" in upper or "LINE_STYLE" in upper) and "LAYER" not in upper:
            linetype_attrs = _sxf_linetype_attributes(record)
            name = str(linetype_attrs.pop("name", "") or ident).strip()
            if name:
                linetype_names[ident] = name
                add_document_linetype(doc, name, source_id=ident, **linetype_attrs)
                _mark_sxf_entity(entity, "linetype")
            continue
        if ident and "LAYER" in upper:
            layer_attrs = _sxf_layer_attributes(record)
            name = layer_attrs.pop("name", None) or ident
            layer_names[ident] = name
            add_document_layer(doc, name, source_id=ident, **layer_attrs)
            _mark_sxf_entity(entity, "layer")
            continue
        if ident and ("CARTESIAN_POINT" in upper or "SXF_POINT" in upper):
            xy = _last_coordinate_pair(record)
            if xy is not None:
                points[ident] = xy
                _mark_sxf_entity(entity, "point")
                continue
        layer = _detect_record_layer(record, layer_names)
        linetype = _detect_record_linetype(record, linetype_names)
        style_kwargs = {"layer": layer, "linetype": linetype}
        _update_sxf_entity_style(entity, layer=layer, linetype=linetype)
        if "DIMENSION" in upper or "SXF_DIM" in upper:
            record_points = _record_points(record, points)
            if len(record_points) < 2:
                record_points = _coordinate_pairs(record)
            if len(record_points) >= 2:
                add_document_dimension(doc, record_points[0], record_points[1], text=_record_label(record), source_id=ident, **style_kwargs)
                _mark_sxf_entity(entity, "dimension")
                continue
        if ("TEXT" in upper or "ANNOTATION" in upper or "NOTE" in upper) and "CARTESIAN_POINT" not in upper:
            label = _record_label(record)
            record_points = _record_points(record, points)
            if not record_points:
                record_points = _coordinate_pairs(record)
            if label and record_points:
                add_document_annotation(doc, record_points[-1], label, source_id=ident, **style_kwargs)
                _mark_sxf_entity(entity, "annotation")
                continue
        if any(name in upper for name in ("REGION", "POLYGON", "AREA", "FACE")) and "LAYER" not in upper:
            record_points = _record_points(record, points)
            if not record_points:
                record_points = _coordinate_pairs(record)
            if len(record_points) >= 3:
                item: dict[str, Any] = {"id": ident, "points": [[x, y] for x, y in record_points]}
                item.update({key: value for key, value in style_kwargs.items() if value not in (None, "")})
                doc["regions"].append(item)
                _mark_sxf_entity(entity, "region")
                continue
        if "HATCH" in upper or "FILL" in upper:
            record_points = _record_points(record, points)
            if not record_points:
                record_points = _coordinate_pairs(record)
            if len(record_points) >= 3:
                add_document_hatch(doc, record_points, source_id=ident, pattern=_record_label(record), solid=("FILL" in upper), **style_kwargs)
                _mark_sxf_entity(entity, "hatch")
                continue
        if "CIRCLE" in upper or "ARC" in upper or "ELLIPSE" in upper or "SPLINE" in upper or "BEZIER" in upper:
            curve = _sxf_curve_mapping(record, points)
            if curve:
                add_document_curve(doc, curve, source_id=ident, **style_kwargs)
                _mark_sxf_entity(entity, "curve")
                continue
        if "POLYLINE" in upper or "COMPOSITE_CURVE" in upper or "COMPOSITE" in upper:
            refs = [ref for ref in _record_refs(record) if ref in points]
            if len(refs) >= 2:
                for segment in segments_from_points([points[ref] for ref in refs], closed=False):
                    add_document_line(doc, segment, source_id=ident, **style_kwargs)
                _mark_sxf_entity(entity, "polyline")
                continue
        if "LINE" in upper or "SEGMENT" in upper:
            refs = [ref for ref in _record_refs(record) if ref in points]
            if len(refs) >= 2:
                a = points[refs[-2]]
                b = points[refs[-1]]
                add_document_line(doc, (a[0], a[1], b[0], b[1]), source_id=ident, **style_kwargs)
                _mark_sxf_entity(entity, "line")
                continue
            nums = _numbers(record)
            if len(nums) >= 5 and ident:
                nums = nums[1:]
            if len(nums) >= 4:
                add_document_line(doc, (nums[0], nums[1], nums[2], nums[3]), source_id=ident, **style_kwargs)
                _mark_sxf_entity(entity, "line")
                continue
        if entity is not None and any(token in upper for token in ("COLOR", "COLOUR", "STYLE", "FONT", "IMAGE", "SYMBOL", "ATTRIBUTE", "USER_DEFINED", "GROUP", "SHEET", "DRAWING", "VIEW", "TRANSFORM")):
            _mark_sxf_entity(entity, "metadata")
    _finalize_sxf_entity_coverage(doc)
    if _document_has_content(doc):
        return doc
    return document_from_lines(parse_plain_lines(text), fmt="plain")


def split_sxf_step_records(text: str) -> list[str]:
    """Split SXF/P21 STEP text on record terminators without breaking quoted strings."""
    records: list[str] = []
    buf: list[str] = []
    in_string = False
    i = 0
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    while i < len(normalized):
        ch = normalized[i]
        if ch == "'":
            buf.append(ch)
            if in_string and i + 1 < len(normalized) and normalized[i + 1] == "'":
                buf.append(normalized[i + 1])
                i += 2
                continue
            in_string = not in_string
            i += 1
            continue
        if ch == ";" and not in_string:
            record = "".join(buf).strip()
            if record:
                records.append(record)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        records.append(tail)
    return records


def export_sxf_document(doc: Mapping[str, Any], *, preserve_raw: bool = True) -> str:
    """Export a CAD document to a lightweight SXF/P21 exchange text.

    If the document came from SXF/P21 and raw STEP entities are present, the
    default path preserves those records verbatim for loss-minimized re-export.
    Generated records are used for internal geometry or when preserve_raw=False.
    """
    raw_entities = [entity for entity in _as_list(doc.get("entities", [])) if isinstance(entity, Mapping) and str(entity.get("raw", "")).strip()]
    if preserve_raw and raw_entities:
        body = [str(entity["raw"]).strip().rstrip(";") for entity in raw_entities]
        return "ISO-10303-21;\nDATA;\n" + ";\n".join(body) + ";\nENDSEC;\nEND-ISO-10303-21;\n"
    return _generated_sxf_document(doc)


def export_sxf_file(doc: Mapping[str, Any], path: str | Path, *, preserve_raw: bool = True) -> Path:
    target = Path(path)
    target.write_text(export_sxf_document(doc, preserve_raw=preserve_raw), encoding="utf-8")
    return target


def validate_sxf_roundtrip(source_path: str | Path, exported_path: str | Path | None = None, *, preserve_raw: bool = True) -> dict[str, Any]:
    source = Path(source_path)
    original = parse_cad_file_document(source)
    exported_text = export_sxf_document(original, preserve_raw=preserve_raw)
    if exported_path is not None:
        Path(exported_path).write_text(exported_text, encoding="utf-8")
    roundtrip = parse_sxf_document(exported_text)
    original_lines = cad_document_line_tuples(original)
    roundtrip_lines = cad_document_line_tuples(roundtrip)
    line_mismatches = _line_multiset_delta(original_lines, roundtrip_lines)
    attr_report = _cad_attribute_roundtrip_report(original, roundtrip)
    original_coverage = original.get("sxf_entity_coverage", {}) if isinstance(original.get("sxf_entity_coverage", {}), Mapping) else {}
    roundtrip_coverage = roundtrip.get("sxf_entity_coverage", {}) if isinstance(roundtrip.get("sxf_entity_coverage", {}), Mapping) else {}
    return {
        "source": str(source),
        "exported": "" if exported_path is None else str(exported_path),
        "preserve_raw": preserve_raw,
        "line_count": len(original_lines),
        "roundtrip_line_count": len(roundtrip_lines),
        "line_mismatch_count": line_mismatches,
        "entity_count": int(original_coverage.get("total", len(_as_list(original.get("entities", []))))),
        "roundtrip_entity_count": int(roundtrip_coverage.get("total", len(_as_list(roundtrip.get("entities", []))))),
        "unhandled_entities": int(original_coverage.get("unhandled", 0) or 0),
        "roundtrip_unhandled_entities": int(roundtrip_coverage.get("unhandled", 0) or 0),
        "attributes": attr_report,
        "ok": line_mismatches == 0 and attr_report["missing_layer_count"] == 0 and attr_report["missing_linetype_count"] == 0 and attr_report["missing_entity_raw_count"] == 0,
    }


def validate_dwg_converter_link(path: str | Path, *, converter: str | Sequence[str] | None = None, output_suffix: str | None = None) -> dict[str, Any]:
    source = Path(path)
    try:
        doc = parse_dwg_file_document(source, converter=converter, output_suffix=output_suffix)
        return {
            "source": str(source),
            "ok": True,
            "converted_suffix": doc.get("converted_suffix", ""),
            "line_count": len(cad_document_line_tuples(doc)),
            "layers": len(_as_list(doc.get("layers", []))),
            "entities": len(_as_list(doc.get("entities", []))),
            "error": "",
        }
    except Exception as exc:
        return {
            "source": str(source),
            "ok": False,
            "converted_suffix": output_suffix or "",
            "line_count": 0,
            "layers": 0,
            "entities": 0,
            "error": str(exc),
        }


def _generated_sxf_document(doc: Mapping[str, Any]) -> str:
    document = document_from_mapping(doc, fmt="sxf") if not any(key in doc for key in ("lines", "curves", "regions", "layers")) else dict(doc)
    records: list[str] = ["ISO-10303-21", "DATA"]
    next_id = 1
    layer_refs: dict[str, str] = {}
    linetype_refs: dict[str, str] = {}
    point_refs: dict[tuple[float, float], str] = {}

    def new_id() -> str:
        nonlocal next_id
        ident = f"#{next_id}"
        next_id += 1
        return ident

    def ref_or_dollar(value: Any, mapping: Mapping[str, str]) -> str:
        if value in (None, ""):
            return "$"
        return mapping.get(str(value), "$")

    def point_ref(point: tuple[float, float]) -> str:
        key = (round(float(point[0]), 12), round(float(point[1]), 12))
        if key not in point_refs:
            ident = new_id()
            point_refs[key] = ident
            records.append(f"{ident}=CARTESIAN_POINT('',({_fmt_float(key[0])},{_fmt_float(key[1])}))")
        return point_refs[key]

    for linetype in _as_list(document.get("linetypes", [])):
        if not isinstance(linetype, Mapping):
            continue
        name = str(linetype.get("name", "")).strip()
        if not name:
            continue
        ident = new_id()
        linetype_refs[name] = ident
        pattern = linetype.get("pattern", [])
        pattern_values = ",".join(_fmt_float(float(value)) for value in pattern) if isinstance(pattern, list) else ""
        description = _sxf_quote(str(linetype.get("description", "")))
        records.append(f"{ident}=SXF_LINETYPE({_sxf_quote(name)},{description}{',' if pattern_values else ''}{pattern_values})")
    for layer in _as_list(document.get("layers", [])):
        if not isinstance(layer, Mapping):
            continue
        name = str(layer.get("name", "")).strip()
        if not name:
            continue
        ident = new_id()
        layer_refs[name] = ident
        records.append(
            f"{ident}=SXF_LAYER({_sxf_quote(name)},{_sxf_quote(str(layer.get('color', '')))},"
            f"{_sxf_quote(str(layer.get('linetype', '')))},"
            f"{_sxf_quote(str(layer.get('parent', layer.get('source', ''))))})"
        )
    for line_index, line in enumerate(_as_list(document.get("lines", [])), start=1):
        if not isinstance(line, Mapping):
            continue
        start = _xy_pair(line.get("start", line.get("p1")))
        end = _xy_pair(line.get("end", line.get("p2")))
        if start is None or end is None:
            continue
        ident = new_id()
        p1 = point_ref(start)
        p2 = point_ref(end)
        layer = ref_or_dollar(line.get("layer"), layer_refs)
        ltype = ref_or_dollar(line.get("linetype"), linetype_refs)
        label = _sxf_quote(str(line.get("id", line.get("source_id", f"line_{line_index}"))))
        records.append(f"{ident}=SXF_LINE_FEATURE({label},{layer},{ltype},{p1},{p2})")
    for curve_index, curve in enumerate(_as_list(document.get("curves", [])), start=1):
        if not isinstance(curve, Mapping):
            continue
        ident = new_id()
        layer = ref_or_dollar(curve.get("layer"), layer_refs)
        label = _sxf_quote(str(curve.get("id", curve.get("source_id", f"curve_{curve_index}"))))
        kind = str(curve.get("type", curve.get("kind", "curve"))).lower()
        if kind == "circle" and _xy_pair(curve.get("center")) is not None:
            center = _xy_pair(curve.get("center"))
            radius = float(curve.get("radius", 0.0))
            records.append(f"{ident}=SXF_CIRCLE_FEATURE({label},{layer},({_fmt_float(center[0])},{_fmt_float(center[1])}),{_fmt_float(radius)})")
        elif kind == "arc" and _xy_pair(curve.get("center")) is not None:
            center = _xy_pair(curve.get("center"))
            radius = float(curve.get("radius", 0.0))
            records.append(
                f"{ident}=SXF_ARC_FEATURE({label},{layer},({_fmt_float(center[0])},{_fmt_float(center[1])}),{_fmt_float(radius)},"
                f"{_fmt_float(float(curve.get('start_angle', 0.0)))},{_fmt_float(float(curve.get('end_angle', 360.0)))})"
            )
        else:
            pts = [_xy_pair(point) for point in _as_list(curve.get("points", []))]
            refs = [point_ref(point) for point in pts if point is not None]
            if len(refs) >= 2:
                entity = "SXF_BEZIER_FEATURE" if "bezier" in kind else "SXF_SPLINE_FEATURE"
                records.append(f"{ident}={entity}({label},{layer},{','.join(refs)})")
    for region_index, region in enumerate(_as_list(document.get("regions", [])), start=1):
        if not isinstance(region, Mapping):
            continue
        refs = [point_ref(point) for point in (_xy_pair(raw) for raw in _as_list(region.get("points", []))) if point is not None]
        if len(refs) >= 3:
            ident = new_id()
            layer = ref_or_dollar(region.get("layer"), layer_refs)
            label = _sxf_quote(str(region.get("id", region.get("source_id", f"region_{region_index}"))))
            records.append(f"{ident}=SXF_POLYGON_FEATURE({label},{layer},{','.join(refs)})")
    for text_index, note in enumerate(_as_list(document.get("annotations", [])), start=1):
        if not isinstance(note, Mapping):
            continue
        point = _xy_pair(note.get("point", note.get("position")))
        if point is None:
            continue
        ident = new_id()
        layer = ref_or_dollar(note.get("layer"), layer_refs)
        label = _sxf_quote(str(note.get("text", note.get("label", f"text_{text_index}"))))
        records.append(f"{ident}=SXF_TEXT({label},{layer},{point_ref(point)})")
    for dim_index, dimension in enumerate(_as_list(document.get("dimensions", [])), start=1):
        if not isinstance(dimension, Mapping):
            continue
        start = _xy_pair(dimension.get("start", dimension.get("p1")))
        end = _xy_pair(dimension.get("end", dimension.get("p2")))
        if start is None or end is None:
            continue
        ident = new_id()
        layer = ref_or_dollar(dimension.get("layer"), layer_refs)
        label = _sxf_quote(str(dimension.get("text", dimension.get("label", f"dim_{dim_index}"))))
        records.append(f"{ident}=SXF_DIMENSION({label},{layer},{point_ref(start)},{point_ref(end)})")
    for hatch_index, hatch in enumerate(_as_list(document.get("hatches", [])), start=1):
        if not isinstance(hatch, Mapping):
            continue
        rings = _hatch_rings_from_mapping(hatch)
        if not rings:
            continue
        ident = new_id()
        layer = ref_or_dollar(hatch.get("layer"), layer_refs)
        label = _sxf_quote(str(hatch.get("pattern", hatch.get("id", f"hatch_{hatch_index}"))))
        refs = [point_ref(point) for point in rings[0]]
        records.append(f"{ident}=SXF_HATCH({label},{layer},{','.join(refs)})")
    records.extend(["ENDSEC", "END-ISO-10303-21"])
    return ";\n".join(records) + ";\n"


def _sxf_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _fmt_float(value: float) -> str:
    text = f"{float(value):.12g}"
    return "0" if text == "-0" else text


def _line_multiset_delta(a: list[Line2D], b: list[Line2D], *, tol: float = 1.0e-8) -> int:
    def key(line: Line2D) -> tuple[int, int, int, int]:
        return tuple(int(round(float(value) / tol)) for value in line)

    unmatched = [key(line) for line in b]
    missing = 0
    for line in a:
        direct = key(line)
        reverse = key((line[2], line[3], line[0], line[1]))
        try:
            unmatched.remove(direct)
        except ValueError:
            try:
                unmatched.remove(reverse)
            except ValueError:
                missing += 1
    return missing + len(unmatched)


def _cad_attribute_roundtrip_report(original: Mapping[str, Any], roundtrip: Mapping[str, Any]) -> dict[str, Any]:
    original_layers = {str(layer.get("name", "")) for layer in _as_list(original.get("layers", [])) if isinstance(layer, Mapping) and layer.get("name")}
    roundtrip_layers = {str(layer.get("name", "")) for layer in _as_list(roundtrip.get("layers", [])) if isinstance(layer, Mapping) and layer.get("name")}
    original_linetypes = {str(item.get("name", "")) for item in _as_list(original.get("linetypes", [])) if isinstance(item, Mapping) and item.get("name")}
    roundtrip_linetypes = {str(item.get("name", "")) for item in _as_list(roundtrip.get("linetypes", [])) if isinstance(item, Mapping) and item.get("name")}
    original_raw = {str(item.get("id")) for item in _as_list(original.get("entities", [])) if isinstance(item, Mapping) and item.get("raw")}
    roundtrip_raw = {str(item.get("id")) for item in _as_list(roundtrip.get("entities", [])) if isinstance(item, Mapping) and item.get("raw")}
    return {
        "missing_layers": sorted(original_layers - roundtrip_layers),
        "missing_layer_count": len(original_layers - roundtrip_layers),
        "missing_linetypes": sorted(original_linetypes - roundtrip_linetypes),
        "missing_linetype_count": len(original_linetypes - roundtrip_linetypes),
        "missing_entity_raw_ids": sorted(original_raw - roundtrip_raw),
        "missing_entity_raw_count": len(original_raw - roundtrip_raw),
    }


def parse_gf1_lines(text: str) -> list[Line2D]:
    return cad_document_line_tuples(parse_gf1_document(text))


def parse_gf1_lines_bytes(data: bytes) -> list[Line2D]:
    return cad_document_line_tuples(parse_gf1_document_bytes(data))


def parse_gf1_document(text: str) -> CadDocument:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return empty_cad_document("gf1")
    if not isinstance(data, Mapping):
        return empty_cad_document("gf1")
    return document_from_mapping(data, fmt="gf1")


def parse_gf1_document_bytes(data: bytes) -> CadDocument:
    for label, text in _gf1_payload_text_candidates(data):
        doc = parse_gf1_document(text)
        if _document_has_content(doc):
            doc["payload_encoding"] = label
            doc["binary_payload"] = _gf1_binary_payload_marker(data, label)
            return doc
    doc = empty_cad_document("gf1")
    doc["binary_payload"] = _looks_binary(data)
    return doc


def empty_cad_document(fmt: str) -> CadDocument:
    return {
        "format": fmt,
        "lines": [],
        "curves": [],
        "regions": [],
        "tunnels": [],
        "layers": [],
        "linetypes": [],
        "annotations": [],
        "dimensions": [],
        "dimension_styles": [],
        "hatches": [],
        "entities": [],
    }


def document_from_lines(lines: list[Line2D], *, fmt: str) -> CadDocument:
    doc = empty_cad_document(fmt)
    for line in lines:
        add_document_line(doc, line)
    return doc


def document_from_mapping(data: Mapping[str, Any], *, fmt: str) -> CadDocument:
    root = data.get("geometry", data)
    doc = empty_cad_document(fmt)
    if not isinstance(root, Mapping):
        return doc
    for linetype in _as_list(root.get("linetypes", root.get("line_types", data.get("linetypes", data.get("line_types", []))))):
        if isinstance(linetype, Mapping):
            name = str(linetype.get("name", linetype.get("id", ""))).strip()
            if name:
                add_document_linetype(doc, name, source_id=linetype.get("id"), **_linetype_kwargs_from_mapping(linetype))
        elif linetype is not None:
            add_document_linetype(doc, str(linetype).strip())
    for layer in _as_list(root.get("layers", data.get("layers", []))):
        if isinstance(layer, Mapping):
            add_document_layer(doc, str(layer.get("name", layer.get("id", ""))).strip(), source_id=layer.get("id"), **_layer_kwargs_from_mapping(layer))
        elif layer is not None:
            add_document_layer(doc, str(layer).strip())
    for item in _as_list(root.get("lines", root.get("linework", []))):
        if not isinstance(item, Mapping):
            continue
        start = _xy_pair(item.get("start", item.get("p1")))
        end = _xy_pair(item.get("end", item.get("p2")))
        if start is not None and end is not None:
            add_document_line(doc, (start[0], start[1], end[0], end[1]), source_id=item.get("id"), **_style_kwargs_from_mapping(item))
    raw_curves = [
        *_as_list(root.get("curves", [])),
        *_as_list(root.get("arcs", [])),
        *_as_list(root.get("splines", [])),
        *_as_list(root.get("beziers", [])),
    ]
    for curve in raw_curves:
        if not isinstance(curve, Mapping):
            continue
        add_document_curve(doc, curve, source_id=curve.get("id"), **_curve_style_kwargs_from_mapping(curve))
    for region in _as_list(root.get("regions", [])):
        if not isinstance(region, Mapping):
            continue
        points = _ring_points_from_mapping(region)
        if len(points) >= 3:
            item: dict[str, Any] = {"id": region.get("id"), "points": [[x, y] for x, y in points]}
            curve_boundary = _curve_parameter_chain_from_value(region.get("curve_boundary", region.get("boundary", region.get("segments", region.get("curves", region.get("edges"))))))
            if curve_boundary:
                item["curve_boundary"] = curve_boundary
            holes: list[list[tuple[float, float]]] = []
            curve_holes: list[list[dict[str, Any]]] = []
            raw_holes = region.get("holes", region.get("islands", []))
            if isinstance(raw_holes, list):
                for raw_hole in raw_holes:
                    hole = _ring_points_from_value(raw_hole)
                    if len(hole) >= 3:
                        holes.append(hole)
                    hole_curve = _curve_parameter_chain_from_value(raw_hole)
                    if hole_curve:
                        curve_holes.append(hole_curve)
            if not holes:
                rings = _hatch_rings_from_mapping(region)
                if len(rings) > 1:
                    holes.extend(rings[1:])
            if holes:
                item["holes"] = [[[x, y] for x, y in hole] for hole in holes]
            if curve_holes:
                item["curve_holes"] = curve_holes
            item.update(_style_kwargs_from_mapping(region))
            doc["regions"].append(item)
    for tunnel in _as_list(root.get("tunnels", root.get("circles", []))):
        if not isinstance(tunnel, Mapping):
            continue
        center = _xy_pair(tunnel.get("center"))
        try:
            radius = float(tunnel.get("radius", tunnel.get("r", 0.0)))
        except (TypeError, ValueError):
            continue
        if center is not None and radius > 0.0:
            item = {"id": tunnel.get("id"), "center": [center[0], center[1]], "radius": radius}
            item.update(_style_kwargs_from_mapping(tunnel))
            doc["tunnels"].append(item)
    for note in _as_list(root.get("annotations", root.get("texts", root.get("notes", [])))):
        if not isinstance(note, Mapping):
            continue
        point = _xy_pair(note.get("point", note.get("position", note.get("at"))))
        text = str(note.get("text", note.get("label", ""))).strip()
        if point is not None and text:
            add_document_annotation(doc, point, text, source_id=note.get("id"), **_style_kwargs_from_mapping(note))
    for dimension in _as_list(root.get("dimensions", root.get("dimension_lines", []))):
        if not isinstance(dimension, Mapping):
            continue
        start = _xy_pair(dimension.get("start", dimension.get("p1")))
        end = _xy_pair(dimension.get("end", dimension.get("p2")))
        if start is not None and end is not None:
            add_document_dimension(doc, start, end, text=dimension.get("text", dimension.get("label")), source_id=dimension.get("id"), **_dimension_kwargs_from_mapping(dimension))
    for style in _as_list(root.get("dimension_styles", root.get("dimstyles", []))):
        if not isinstance(style, Mapping):
            continue
        name = str(style.get("name", style.get("id", ""))).strip()
        if name:
            add_document_dimension_style(doc, name, **_dimension_style_kwargs_from_mapping(style))
    raw_hatches = [*_as_list(root.get("hatches", [])), *_as_list(root.get("fills", []))]
    for hatch in raw_hatches:
        if not isinstance(hatch, Mapping):
            continue
        rings = _hatch_rings_from_mapping(hatch)
        if rings:
            add_document_hatch(doc, rings=rings, source_id=hatch.get("id"), **_hatch_kwargs_from_mapping(hatch))
    return doc


def cad_document_line_tuples(doc: Mapping[str, Any]) -> list[Line2D]:
    out: list[Line2D] = []
    for item in _as_list(doc.get("lines", [])):
        if isinstance(item, Mapping):
            start = _xy_pair(item.get("start", item.get("p1")))
            end = _xy_pair(item.get("end", item.get("p2")))
            if start is not None and end is not None:
                out.append((start[0], start[1], end[0], end[1]))
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                out.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
            except (TypeError, ValueError):
                pass
    for region in _as_list(doc.get("regions", [])):
        if isinstance(region, Mapping):
            points = [_xy_pair(value) for value in _as_list(region.get("points", [])) if _xy_pair(value) is not None]
            out.extend(segments_from_points(points, closed=True))
            for hole in _as_list(region.get("holes", [])):
                hole_points = [_xy_pair(value) for value in _as_list(hole) if _xy_pair(value) is not None]
                out.extend(segments_from_points(hole_points, closed=True))
    for tunnel in _as_list(doc.get("tunnels", [])):
        if not isinstance(tunnel, Mapping):
            continue
        center = _xy_pair(tunnel.get("center"))
        try:
            radius = float(tunnel.get("radius", 0.0))
        except (TypeError, ValueError):
            continue
        if center is not None and radius > 0.0:
            out.extend(arc_segments(center[0], center[1], radius, 0.0, 360.0, full_circle=True))
    for hatch in _as_list(doc.get("hatches", [])):
        if isinstance(hatch, Mapping):
            rings = _hatch_rings_from_mapping(hatch)
            for ring in rings:
                out.extend(segments_from_points(ring, closed=True))
    return out


def add_document_layer(
    doc: CadDocument,
    name: str,
    *,
    source_id: Any = None,
    color: Any = None,
    linetype: Any = None,
    linetype_pattern: Any = None,
    parent: Any = None,
    visible: Any = None,
    lineweight: Any = None,
    aci_color: Any = None,
) -> None:
    if not name:
        return
    existing = next((layer for layer in doc["layers"] if isinstance(layer, dict) and str(layer.get("name")) == name), None)
    if existing is not None:
        for key, value in {
            "color": color,
            "linetype": linetype,
            "linetype_pattern": normalize_linetype_pattern(linetype_pattern),
            "parent": parent,
            "visible": visible,
            "lineweight": lineweight,
            "aci_color": aci_color,
        }.items():
            if value not in (None, "") and key not in existing:
                existing[key] = value
        return
    item: dict[str, Any] = {"name": name}
    if source_id is not None:
        item["source_id"] = str(source_id)
    for key, value in {
        "color": color,
        "linetype": linetype,
        "linetype_pattern": normalize_linetype_pattern(linetype_pattern),
        "parent": parent,
        "visible": visible,
        "lineweight": lineweight,
        "aci_color": aci_color,
    }.items():
        if value not in (None, ""):
            item[key] = value
    doc["layers"].append(item)


def add_document_linetype(
    doc: CadDocument,
    name: str,
    *,
    source_id: Any = None,
    pattern: Any = None,
    description: Any = None,
    pattern_length: Any = None,
) -> None:
    name = str(name).strip()
    if not name:
        return
    item: dict[str, Any] = {"name": name}
    if source_id is not None:
        item["source_id"] = str(source_id)
    normalized = normalize_linetype_pattern(pattern)
    if normalized:
        item["pattern"] = normalized
    if description not in (None, ""):
        item["description"] = str(description)
    if pattern_length not in (None, ""):
        try:
            item["pattern_length"] = float(pattern_length)
        except (TypeError, ValueError):
            pass
    existing = next((linetype for linetype in doc["linetypes"] if isinstance(linetype, dict) and str(linetype.get("name")) == name), None)
    if existing is not None:
        existing.update({key: value for key, value in item.items() if value not in (None, "", [])})
        return
    doc["linetypes"].append(item)


def add_document_line(
    doc: CadDocument,
    line: Line2D,
    *,
    layer: str | None = None,
    source_id: Any = None,
    color: Any = None,
    linetype: Any = None,
    linetype_pattern: Any = None,
    lineweight: Any = None,
) -> None:
    item: dict[str, Any] = {"start": [float(line[0]), float(line[1])], "end": [float(line[2]), float(line[3])]}
    if layer:
        item["layer"] = layer
        add_document_layer(doc, layer)
    if source_id is not None:
        item["source_id"] = str(source_id)
    for key, value in {"color": color, "linetype": linetype, "linetype_pattern": normalize_linetype_pattern(linetype_pattern), "lineweight": lineweight}.items():
        if value not in (None, ""):
            item[key] = value
    doc["lines"].append(item)


def add_document_curve(
    doc: CadDocument,
    curve: Mapping[str, Any],
    *,
    source_id: Any = None,
    layer: str | None = None,
    color: Any = None,
    linetype: Any = None,
    linetype_pattern: Any = None,
    lineweight: Any = None,
) -> None:
    points = curve_points_from_mapping(curve)
    if len(points) < 2:
        return
    curve_id = source_id if source_id is not None else curve.get("id")
    kind = str(curve.get("type", curve.get("kind", "curve"))).strip().lower()
    closed = kind in {"circle", "ellipse"} or bool(curve.get("closed", curve.get("full", False)))
    item: dict[str, Any] = {
        "id": curve_id,
        "type": kind or "curve",
        "points": [[float(x), float(y)] for x, y in points],
        "segment_count": len(points) if closed and len(points) > 2 else max(len(points) - 1, 0),
    }
    params = _curve_parameter_mapping(curve)
    if params:
        item["parameters"] = params
        for key, value in params.items():
            if key not in item and value not in (None, "", []):
                item[key] = value
    if closed:
        item["closed"] = True
    if layer:
        item["layer"] = layer
        add_document_layer(doc, layer)
    if curve_id is not None:
        item["source_id"] = str(curve_id)
    for key, value in {"color": color, "linetype": linetype, "linetype_pattern": normalize_linetype_pattern(linetype_pattern), "lineweight": lineweight}.items():
        if value not in (None, ""):
            item[key] = value
    doc["curves"].append(item)
    for segment in segments_from_points(points, closed=closed):
        add_document_line(doc, segment, layer=layer, source_id=curve_id, color=color, linetype=linetype, linetype_pattern=linetype_pattern, lineweight=lineweight)


def add_document_annotation(
    doc: CadDocument,
    point: tuple[float, float],
    text: str,
    *,
    layer: str | None = None,
    source_id: Any = None,
    color: Any = None,
    linetype: Any = None,
    linetype_pattern: Any = None,
    lineweight: Any = None,
) -> None:
    item: dict[str, Any] = {"point": [float(point[0]), float(point[1])], "text": str(text)}
    if layer:
        item["layer"] = layer
        add_document_layer(doc, layer)
    if source_id is not None:
        item["source_id"] = str(source_id)
    for key, value in {"color": color, "linetype": linetype, "linetype_pattern": normalize_linetype_pattern(linetype_pattern), "lineweight": lineweight}.items():
        if value not in (None, ""):
            item[key] = value
    doc["annotations"].append(item)


def add_document_dimension(
    doc: CadDocument,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    text: Any = None,
    layer: str | None = None,
    source_id: Any = None,
    color: Any = None,
    linetype: Any = None,
    linetype_pattern: Any = None,
    lineweight: Any = None,
    dimension_style: Any = None,
) -> None:
    value = math.hypot(end[0] - start[0], end[1] - start[1])
    label = "" if text is None else str(text)
    item: dict[str, Any] = {"start": [float(start[0]), float(start[1])], "end": [float(end[0]), float(end[1])], "value": value}
    if label:
        item["text"] = label
    if layer:
        item["layer"] = layer
        add_document_layer(doc, layer)
    if source_id is not None:
        item["source_id"] = str(source_id)
    for key, value in {"color": color, "linetype": linetype, "linetype_pattern": normalize_linetype_pattern(linetype_pattern), "lineweight": lineweight}.items():
        if value not in (None, ""):
            item[key] = value
    if dimension_style not in (None, ""):
        item["dimension_style"] = dimension_style
    doc["dimensions"].append(item)


def add_document_dimension_style(
    doc: CadDocument,
    name: str,
    *,
    text_height: Any = None,
    arrow_size: Any = None,
    scale: Any = None,
    color: Any = None,
    postfix: Any = None,
) -> None:
    item: dict[str, Any] = {"name": name}
    for key, value in {
        "text_height": text_height,
        "arrow_size": arrow_size,
        "scale": scale,
        "color": color,
        "postfix": postfix,
    }.items():
        if value not in (None, ""):
            item[key] = value
    existing = next((style for style in doc["dimension_styles"] if isinstance(style, dict) and str(style.get("name")) == name), None)
    if existing is not None:
        existing.update({key: value for key, value in item.items() if value not in (None, "")})
        return
    doc["dimension_styles"].append(item)


def add_document_hatch(
    doc: CadDocument,
    points: list[tuple[float, float]] | None = None,
    *,
    rings: list[list[tuple[float, float]]] | None = None,
    layer: str | None = None,
    source_id: Any = None,
    color: Any = None,
    linetype: Any = None,
    linetype_pattern: Any = None,
    lineweight: Any = None,
    pattern: Any = None,
    solid: Any = None,
) -> None:
    normalized_rings = normalize_hatch_rings(rings if rings is not None else ([points] if points is not None else []))
    if not normalized_rings:
        return
    item: dict[str, Any] = {"points": [[float(x), float(y)] for x, y in normalized_rings[0]]}
    if len(normalized_rings) > 1:
        item["rings"] = [[[float(x), float(y)] for x, y in ring] for ring in normalized_rings]
        item["island_count"] = len(normalized_rings) - 1
    if layer:
        item["layer"] = layer
        add_document_layer(doc, layer)
    if source_id is not None:
        item["source_id"] = str(source_id)
    for key, value in {
        "color": color,
        "linetype": linetype,
        "linetype_pattern": normalize_linetype_pattern(linetype_pattern),
        "lineweight": lineweight,
        "pattern": pattern,
        "solid": solid,
    }.items():
        if value not in (None, ""):
            item[key] = value
    doc["hatches"].append(item)


def _document_has_content(doc: Mapping[str, Any]) -> bool:
    return any(_as_list(doc.get(key, [])) for key in ("lines", "curves", "regions", "tunnels", "annotations", "dimensions", "dimension_styles", "hatches", "layers", "linetypes", "entities"))


def _gf1_payload_text_candidates(data: bytes) -> list[tuple[str, str]]:
    return _gf1_payload_text_candidates_core(data)


def _gf1_payload_blobs(data: bytes) -> list[tuple[str, bytes]]:
    return _gf1_payload_blobs_core(data)


def _zlib_payloads(data: bytes) -> list[tuple[int, bytes]]:
    return _zlib_payloads_core(data)


def _decoded_text_candidates(data: bytes) -> list[tuple[str, str]]:
    return _decoded_text_candidates_core(data)


def _decode_with_encoding(data: bytes, encoding: str) -> str:
    return _decode_with_encoding_core(data, encoding)


def _gf1_text_candidates(text: str) -> list[tuple[str, str]]:
    return _gf1_text_candidates_core(text)


def _json_payloads(text: str) -> list[tuple[int, str]]:
    return _json_payloads_core(text)


def _looks_like_gf1_text(text: str) -> bool:
    return _looks_like_gf1_text_core(text)


def _best_effort_decode(data: bytes) -> str:
    return _best_effort_decode_core(data)


def _gf1_binary_payload_marker(data: bytes, label: str) -> bool:
    return _gf1_binary_payload_marker_core(data, label)


def _looks_binary(data: bytes) -> bool:
    return _looks_binary_core(data)


def lines_from_mapping(data: Mapping[str, Any]) -> list[Line2D]:
    root = data.get("geometry", data)
    if not isinstance(root, Mapping):
        return []
    out: list[Line2D] = []
    raw_lines = root.get("lines", root.get("linework", []))
    if isinstance(raw_lines, list):
        for item in raw_lines:
            if not isinstance(item, Mapping):
                continue
            start = _xy_pair(item.get("start", item.get("p1")))
            end = _xy_pair(item.get("end", item.get("p2")))
            if start is not None and end is not None:
                out.append((start[0], start[1], end[0], end[1]))
    raw_regions = root.get("regions", [])
    if isinstance(raw_regions, list):
        for region in raw_regions:
            if not isinstance(region, Mapping):
                continue
            points = _ring_points_from_mapping(region)
            out.extend(segments_from_points(points, closed=True))
            rings = _hatch_rings_from_mapping(region)
            for ring in rings[1:]:
                out.extend(segments_from_points(ring, closed=True))
    raw_tunnels = root.get("tunnels", root.get("circles", []))
    if isinstance(raw_tunnels, list):
        for tunnel in raw_tunnels:
            if not isinstance(tunnel, Mapping):
                continue
            center = _xy_pair(tunnel.get("center"))
            try:
                radius = float(tunnel.get("radius", tunnel.get("r", 0.0)))
            except (TypeError, ValueError):
                continue
            if center is not None and radius > 0.0:
                out.extend(arc_segments(center[0], center[1], radius, 0.0, 360.0, full_circle=True))
    return out


def segments_from_points(points: list[tuple[float, float]], *, closed: bool) -> list[Line2D]:
    if len(points) < 2:
        return []
    segments = [(a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:])]
    if closed and len(points) > 2:
        a = points[-1]
        b = points[0]
        segments.append((a[0], a[1], b[0], b[1]))
    return segments


def arc_segments(cx: float, cy: float, radius: float, start_deg: float, end_deg: float, *, full_circle: bool) -> list[Line2D]:
    points = arc_points(cx, cy, radius, start_deg, end_deg, full_circle=full_circle)
    return segments_from_points(points, closed=full_circle)


def arc_points(cx: float, cy: float, radius: float, start_deg: float, end_deg: float, *, full_circle: bool, segments: int | None = None) -> list[tuple[float, float]]:
    if radius <= 0.0:
        return []
    if full_circle:
        start_deg = 0.0
        end_deg = 360.0
    while end_deg <= start_deg:
        end_deg += 360.0
    sweep = end_deg - start_deg
    count = max(4, int(segments or math.ceil(abs(sweep) / 12.0)))
    if full_circle:
        count = max(16, count)
    else:
        count = max(4, count)
    points = []
    for i in range(count + 1):
        angle = math.radians(start_deg + sweep * i / count)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    if full_circle and points and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= 1.0e-12:
        points.pop()
    return points


def ellipse_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    start_deg: float,
    end_deg: float,
    *,
    rotation_deg: float = 0.0,
    full_ellipse: bool = False,
    segments: int | None = None,
) -> list[tuple[float, float]]:
    if rx <= 0.0 or ry <= 0.0:
        return []
    if full_ellipse:
        start_deg = 0.0
        end_deg = 360.0
    while end_deg <= start_deg:
        end_deg += 360.0
    sweep = end_deg - start_deg
    count = max(4, int(segments or math.ceil(abs(sweep) / 12.0)))
    if full_ellipse:
        count = max(16, count)
    theta = math.radians(rotation_deg)
    ct = math.cos(theta)
    st = math.sin(theta)
    points: list[tuple[float, float]] = []
    for i in range(count + 1):
        angle = math.radians(start_deg + sweep * i / count)
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        points.append((cx + x * ct - y * st, cy + x * st + y * ct))
    if full_ellipse and points and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= 1.0e-12:
        points.pop()
    return points


def _assign_float(target: dict[str, float], key: str, raw: str) -> None:
    value = _float_or_none(raw)
    if value is not None:
        target[key] = value


def _assign_dxf_entity_attr(target: dict[str, Any], code: str, raw: str) -> None:
    if code == "8":
        target["layer"] = raw
    elif code == "6":
        target["linetype"] = raw
    elif code == "62":
        target["color"] = dxf_color_name(raw)
        target["aci_color"] = _int_or_none(raw)
    elif code == "370":
        target["lineweight"] = _int_or_none(raw)


def _style_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("layer", "color", "linetype", "lineweight"):
        raw = value.get(key, value.get("line_type" if key == "linetype" else key))
        if raw not in (None, ""):
            out[key] = raw
    pattern = _linetype_pattern_from_mapping(value)
    if pattern:
        out["linetype_pattern"] = pattern
    return out


def _curve_style_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out = _style_kwargs_from_mapping(value)
    if "segments" in value and not any(key in value for key in ("linetype_pattern", "dash_pattern", "line_pattern", "pattern_segments")):
        out.pop("linetype_pattern", None)
    return out


def _dimension_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out = _style_kwargs_from_mapping(value)
    raw = value.get("dimension_style", value.get("dimstyle", value.get("style")))
    if raw not in (None, ""):
        out["dimension_style"] = raw
    return out


def _dimension_style_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("text_height", "arrow_size", "scale", "color", "postfix"):
        raw = value.get(key)
        if raw not in (None, ""):
            out[key] = raw
    return out


def _hatch_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out = _style_kwargs_from_mapping(value)
    for key in ("pattern", "solid"):
        raw = value.get(key)
        if raw not in (None, ""):
            out[key] = raw
    return out


def _linetype_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pattern = normalize_linetype_pattern(value.get("pattern")) or _linetype_pattern_from_mapping(value)
    if pattern:
        out["pattern"] = pattern
    for key in ("description", "pattern_length"):
        raw = value.get(key, value.get("length" if key == "pattern_length" else key))
        if raw not in (None, ""):
            out[key] = raw
    return out


def _hatch_rings_from_mapping(value: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    raw_rings = value.get("rings", value.get("boundaries"))
    rings: list[list[tuple[float, float]]] = []
    if isinstance(raw_rings, list):
        for raw_ring in raw_rings:
            ring = _ring_points_from_value(raw_ring)
            if len(ring) >= 3:
                rings.append(ring)
    if not rings:
        ring = _ring_points_from_mapping(value)
        if len(ring) >= 3:
            rings.append(ring)
    raw_holes = value.get("holes", value.get("islands", []))
    if isinstance(raw_holes, list):
        for raw_hole in raw_holes:
            hole = _ring_points_from_value(raw_hole)
            if len(hole) >= 3:
                rings.append(hole)
    return rings


def _ring_points_from_mapping(value: Mapping[str, Any]) -> list[tuple[float, float]]:
    for key in ("points", "boundary", "segments", "curves", "edges"):
        if key in value:
            points = _ring_points_from_value(value.get(key))
            if len(points) >= 2:
                return points
    return []


def _ring_points_from_value(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        return _dedupe_curve_points(curve_points_from_mapping(value), closed=True)
    if not isinstance(value, list):
        return []
    direct = [_xy_pair(item) for item in value if _xy_pair(item) is not None]
    if len(direct) == len(value) and direct:
        return _dedupe_curve_points(direct, closed=True)
    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            segment_points = curve_points_from_mapping(item)
        else:
            xy = _xy_pair(item)
            segment_points = [xy] if xy is not None else []
        _append_curve_points(points, segment_points)
    return _dedupe_curve_points(points, closed=True)


def _curve_parameter_chain_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        spec = _curve_parameter_mapping(value)
        if spec:
            return [spec]
        for key in ("curve_boundary", "boundary", "segments", "curves", "edges"):
            specs = _curve_parameter_chain_from_value(value.get(key))
            if specs:
                return specs
        return []
    if not isinstance(value, list):
        return []
    specs: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            spec = _curve_parameter_mapping(item)
            if spec:
                specs.append(spec)
    return specs


def _curve_parameter_mapping(value: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = _curve_type(value)
    if not kind and "start" in value and "end" in value:
        kind = "line"
    if kind in {"line", "segment"}:
        start = _xy_pair(value.get("start", value.get("p1")))
        end = _xy_pair(value.get("end", value.get("p2")))
        if start is None or end is None:
            return None
        return {"type": "line", "start": [float(start[0]), float(start[1])], "end": [float(end[0]), float(end[1])]}
    if kind in {"nurbs", "nurbs_curve", "rational_bspline", "rational_b_spline"} or (kind in {"spline", "bspline", "b_spline"} and any(key in value for key in ("knots", "weights", "degree", "order"))):
        controls = [_xy_pair(point) for point in _as_list(value.get("control_points", value.get("points", []))) if _xy_pair(point) is not None]
        if len(controls) < 2:
            return None
        degree_raw = value.get("degree", value.get("order", min(3, len(controls) - 1)))
        try:
            degree = int(degree_raw)
        except (TypeError, ValueError):
            degree = min(3, len(controls) - 1)
        if "order" in value and "degree" not in value:
            degree -= 1
        degree = max(1, min(degree, len(controls) - 1))
        weights = _float_list(value.get("weights"))
        if len(weights) != len(controls):
            weights = [1.0] * len(controls)
        knots = _float_list(value.get("knots", value.get("knot_vector")))
        if len(knots) != len(controls) + degree + 1:
            knots = _open_uniform_knot_vector(len(controls), degree)
        return {
            "type": "nurbs",
            "control_points": [[float(x), float(y)] for x, y in controls],
            "weights": weights,
            "knots": knots,
            "degree": degree,
            "closed": bool(value.get("closed", value.get("periodic", False))),
        }
    if kind in {"bezier", "spline", "bspline", "b_spline"}:
        controls = [_xy_pair(point) for point in _as_list(value.get("control_points", value.get("points", []))) if _xy_pair(point) is not None]
        if len(controls) < 2:
            return None
        return {"type": "bezier", "control_points": [[float(x), float(y)] for x, y in controls]}
    center = _xy_pair(value.get("center", value.get("origin")))
    if center is None:
        return None
    rx = _float_value(value, "rx", "radius_x", "major_radius", "a")
    ry = _float_value(value, "ry", "radius_y", "minor_radius", "b")
    radius = _float_value(value, "radius", "r")
    if kind in {"ellipse", "elliptic_arc"} or rx is not None or ry is not None:
        rx = rx if rx is not None else radius
        ry = ry if ry is not None else rx
        if rx is None or ry is None or rx <= 0.0 or ry <= 0.0:
            return None
        closed = kind == "ellipse" or bool(value.get("closed", value.get("full", False)))
        return {
            "type": "ellipse" if closed else "elliptic_arc",
            "center": [float(center[0]), float(center[1])],
            "rx": float(rx),
            "ry": float(ry),
            "start_angle": _angle_value(value, "start_angle", "start_deg", "angle_start", default=0.0),
            "end_angle": _angle_value(value, "end_angle", "end_deg", "angle_end", default=360.0),
            "rotation": _angle_value(value, "rotation", "rotation_deg", default=0.0),
            "closed": closed,
        }
    if radius is None or radius <= 0.0:
        return None
    closed = kind == "circle" or bool(value.get("closed", value.get("full", False)))
    return {
        "type": "circle" if closed else "arc",
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "start_angle": _angle_value(value, "start_angle", "start_deg", "angle_start", default=0.0),
        "end_angle": _angle_value(value, "end_angle", "end_deg", "angle_end", default=360.0),
        "closed": closed,
    }


def curve_points_from_mapping(value: Mapping[str, Any]) -> list[tuple[float, float]]:
    nested = value.get("points")
    if nested is not None and not _curve_type(value):
        return _ring_points_from_value(nested)
    if any(key in value for key in ("boundary", "segments", "curves", "edges")) and not _curve_type(value):
        return _ring_points_from_mapping(value)
    kind = _curve_type(value)
    if kind in {"line", "segment"} or ("start" in value and "end" in value and "radius" not in value and "center" not in value):
        start = _xy_pair(value.get("start", value.get("p1")))
        end = _xy_pair(value.get("end", value.get("p2")))
        return [point for point in (start, end) if point is not None]
    if kind in {"nurbs", "nurbs_curve", "rational_bspline", "rational_b_spline"} or (kind in {"spline", "bspline", "b_spline"} and any(key in value for key in ("knots", "weights", "degree", "order"))):
        controls = [_xy_pair(point) for point in _as_list(value.get("control_points", value.get("points", []))) if _xy_pair(point) is not None]
        if len(controls) < 2:
            return []
        degree_raw = value.get("degree", value.get("order", min(3, len(controls) - 1)))
        try:
            degree = int(degree_raw)
        except (TypeError, ValueError):
            degree = min(3, len(controls) - 1)
        if "order" in value and "degree" not in value:
            degree -= 1
        degree = max(1, min(degree, len(controls) - 1))
        weights = _float_list(value.get("weights"))
        if len(weights) != len(controls):
            weights = [1.0] * len(controls)
        knots = _float_list(value.get("knots", value.get("knot_vector")))
        if len(knots) != len(controls) + degree + 1:
            knots = _open_uniform_knot_vector(len(controls), degree)
        segments = max(4, _int_or_none(value.get("segments")) or len(controls) * 8)
        return [_eval_nurbs(controls, weights, knots, degree, i / segments) for i in range(segments + 1)]
    if kind in {"bezier", "spline", "bspline", "b_spline"}:
        controls = [_xy_pair(point) for point in _as_list(value.get("control_points", value.get("points", []))) if _xy_pair(point) is not None]
        if len(controls) < 2:
            return []
        segments = max(2, _int_or_none(value.get("segments")) or len(controls) * 8)
        return [_de_casteljau(controls, i / segments) for i in range(segments + 1)]
    center = _xy_pair(value.get("center", value.get("origin")))
    if center is not None:
        segments = _int_or_none(value.get("segments", value.get("division_count")))
        rx = _float_value(value, "rx", "radius_x", "major_radius", "a")
        ry = _float_value(value, "ry", "radius_y", "minor_radius", "b")
        if rx is not None or ry is not None or kind in {"ellipse", "elliptic_arc"}:
            rx = rx if rx is not None else _float_value(value, "radius", "r")
            ry = ry if ry is not None else rx
            if rx is None or ry is None:
                return []
            return ellipse_points(
                center[0],
                center[1],
                rx,
                ry,
                _angle_value(value, "start_angle", "start_deg", "angle_start", default=0.0),
                _angle_value(value, "end_angle", "end_deg", "angle_end", default=360.0),
                rotation_deg=_angle_value(value, "rotation", "rotation_deg", default=0.0),
                full_ellipse=kind in {"ellipse", "circle"} or bool(value.get("full", value.get("closed", False))),
                segments=segments,
            )
        radius = _float_value(value, "radius", "r")
        if radius is not None:
            full = kind == "circle" or bool(value.get("full", value.get("closed", False)))
            return arc_points(
                center[0],
                center[1],
                radius,
                _angle_value(value, "start_angle", "start_deg", "angle_start", default=0.0),
                _angle_value(value, "end_angle", "end_deg", "angle_end", default=360.0),
                full_circle=full,
                segments=segments,
            )
    if nested is not None:
        return _ring_points_from_value(nested)
    return []


def _curve_type(value: Mapping[str, Any]) -> str:
    return str(value.get("type", value.get("kind", value.get("curve_type", "")))).strip().lower()


def _float_value(value: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in value:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                return None
    return None


def _angle_value(value: Mapping[str, Any], *keys: str, default: float) -> float:
    parsed = _float_value(value, *keys)
    return default if parsed is None else parsed


def _float_list(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _open_uniform_knot_vector(control_count: int, degree: int) -> list[float]:
    if control_count <= 1:
        return [0.0, 1.0]
    degree = max(1, min(degree, control_count - 1))
    interior = control_count - degree - 1
    knots = [0.0] * (degree + 1)
    if interior > 0:
        knots.extend(index / (interior + 1) for index in range(1, interior + 1))
    knots.extend([1.0] * (degree + 1))
    return knots


def _append_curve_points(target: list[tuple[float, float]], points: list[tuple[float, float]]) -> None:
    for point in points:
        if target and math.hypot(target[-1][0] - point[0], target[-1][1] - point[1]) <= 1.0e-10:
            continue
        target.append(point)


def _dedupe_curve_points(points: list[tuple[float, float]], *, closed: bool) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    _append_curve_points(out, points)
    if closed and len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= 1.0e-10:
        out.pop()
    return out


def _de_casteljau(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    work = [(float(x), float(y)) for x, y in points]
    while len(work) > 1:
        work = [
            (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
            for a, b in zip(work, work[1:])
        ]
    return work[0]


def _eval_nurbs(
    controls: list[tuple[float, float]],
    weights: list[float],
    knots: list[float],
    degree: int,
    t: float,
) -> tuple[float, float]:
    if not controls:
        return (0.0, 0.0)
    degree = max(1, min(degree, len(controls) - 1))
    if len(weights) != len(controls):
        weights = [1.0] * len(controls)
    if len(knots) != len(controls) + degree + 1:
        knots = _open_uniform_knot_vector(len(controls), degree)
    n = len(controls) - 1
    u_min = knots[degree]
    u_max = knots[n + 1]
    if u_max <= u_min:
        return controls[-1] if t >= 1.0 else controls[0]
    u = u_min + min(max(float(t), 0.0), 1.0) * (u_max - u_min)
    if u >= u_max:
        u = u_max - max((u_max - u_min) * 1.0e-12, 1.0e-14)
    numerator_x = 0.0
    numerator_y = 0.0
    denominator = 0.0
    for index, (x, y) in enumerate(controls):
        basis = _nurbs_basis(index, degree, u, knots)
        weighted = basis * weights[index]
        numerator_x += weighted * x
        numerator_y += weighted * y
        denominator += weighted
    if abs(denominator) <= 1.0e-30:
        return controls[-1] if t >= 0.5 else controls[0]
    return numerator_x / denominator, numerator_y / denominator


def _nurbs_basis(index: int, degree: int, u: float, knots: list[float]) -> float:
    if degree == 0:
        return 1.0 if knots[index] <= u < knots[index + 1] else 0.0
    left_den = knots[index + degree] - knots[index]
    right_den = knots[index + degree + 1] - knots[index + 1]
    left = 0.0
    right = 0.0
    if abs(left_den) > 1.0e-30:
        left = (u - knots[index]) / left_den * _nurbs_basis(index, degree - 1, u, knots)
    if abs(right_den) > 1.0e-30:
        right = (knots[index + degree + 1] - u) / right_den * _nurbs_basis(index + 1, degree - 1, u, knots)
    return left + right


def normalize_hatch_rings(raw_rings: list[Any]) -> list[list[tuple[float, float]]]:
    rings: list[list[tuple[float, float]]] = []
    for raw_ring in raw_rings:
        ring: list[tuple[float, float]] = []
        for point in _as_list(raw_ring):
            xy = _xy_pair(point)
            if xy is not None:
                ring.append(xy)
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _layer_kwargs_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out = _style_kwargs_from_mapping(value)
    out.pop("layer", None)
    for key in ("parent", "visible", "aci_color"):
        raw = value.get(key)
        if raw not in (None, ""):
            out[key] = raw
    return out


def _linetype_pattern_from_mapping(value: Mapping[str, Any]) -> list[float] | None:
    for key in ("linetype_pattern", "dash_pattern", "line_pattern", "pattern_segments", "segments"):
        raw = value.get(key)
        pattern = normalize_linetype_pattern(raw)
        if pattern:
            return pattern
    return None


def normalize_linetype_pattern(raw: Any) -> list[float] | None:
    if raw in (None, ""):
        return None
    values: list[Any]
    if isinstance(raw, str):
        values = [part for part in re.split(r"[,\s;]+", raw.strip()) if part]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]
    pattern: list[float] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("length", value.get("value", 0.0))
        try:
            pattern.append(float(value))
        except (TypeError, ValueError):
            continue
    if not pattern or all(abs(value) <= 1.0e-12 for value in pattern):
        return None
    return pattern


def _sxf_layer_attributes(record: str) -> dict[str, Any]:
    labels = [label for label in _quoted_strings(record) if label.strip()]
    attrs: dict[str, Any] = {"name": labels[0]} if labels else {}
    if len(labels) >= 2:
        attrs["color"] = labels[1]
    if len(labels) >= 3:
        attrs["linetype"] = labels[2]
    if len(labels) >= 4:
        attrs["parent"] = labels[3]
    return attrs


def _sxf_linetype_attributes(record: str) -> dict[str, Any]:
    labels = [label for label in _quoted_strings(record) if label.strip()]
    attrs: dict[str, Any] = {"name": labels[0]} if labels else {}
    if len(labels) >= 2:
        label_pattern = normalize_linetype_pattern(labels[-1])
        if label_pattern:
            attrs["pattern"] = label_pattern
            if len(labels) > 2:
                attrs["description"] = labels[1]
        else:
            attrs["description"] = labels[1]
    nums = _numbers(record)
    ident = _record_id(record)
    if ident and nums and str(int(nums[0])) == ident[1:]:
        nums = nums[1:]
    numeric_pattern = normalize_linetype_pattern(nums)
    if numeric_pattern:
        attrs["pattern"] = numeric_pattern
    return attrs


def _sxf_curve_mapping(record: str, points: Mapping[str, tuple[float, float]]) -> dict[str, Any]:
    upper = record.upper()
    refs = [ref for ref in _record_refs(record) if ref in points]
    record_points = [points[ref] for ref in refs]
    coords = _coordinate_pairs(record)
    nums = _numbers_without_record_refs(record)
    if "BEZIER" in upper or "SPLINE" in upper:
        controls = record_points or coords
        if len(controls) < 2:
            return {}
        return {"type": "bezier" if "BEZIER" in upper else "spline", "points": [[x, y] for x, y in controls]}
    if "ELLIPSE" in upper:
        if len(nums) >= 4:
            curve: dict[str, Any] = {"type": "ellipse" if "ARC" not in upper else "elliptic_arc", "center": [nums[0], nums[1]], "rx": nums[2], "ry": nums[3]}
            if len(nums) >= 6:
                curve["start_angle"] = nums[4]
                curve["end_angle"] = nums[5]
            if len(nums) >= 7:
                curve["rotation"] = nums[6]
            return curve
        return {}
    if "CIRCLE" in upper or "ARC" in upper:
        center: tuple[float, float] | None = coords[0] if coords else None
        radius: float | None = None
        start = 0.0
        end = 360.0
        if center is not None:
            tail = nums[2:] if len(nums) >= 2 and abs(nums[0] - center[0]) <= 1.0e-12 and abs(nums[1] - center[1]) <= 1.0e-12 else nums
            if tail:
                radius = tail[0]
                if len(tail) >= 3:
                    start = tail[1]
                    end = tail[2]
        elif len(nums) >= 3:
            center = (nums[0], nums[1])
            radius = nums[2]
            if len(nums) >= 5:
                start = nums[3]
                end = nums[4]
        if center is None or radius is None:
            return {}
        return {"type": "circle" if "CIRCLE" in upper else "arc", "center": [center[0], center[1]], "radius": radius, "start_angle": start, "end_angle": end, "full": "CIRCLE" in upper}
    return {}


def dxf_color_name(raw: Any) -> str | None:
    value = _int_or_none(raw)
    if value is None:
        return None
    table = {
        1: "red",
        2: "yellow",
        3: "green",
        4: "cyan",
        5: "blue",
        6: "magenta",
        7: "white",
        8: "gray",
        9: "lightgray",
    }
    return table.get(abs(value), str(value))


def _dxf_dimension_point(values: Mapping[str, Any], x_key: str, y_key: str) -> tuple[float, float] | None:
    if x_key in values and y_key in values:
        return float(values[x_key]), float(values[y_key])
    return None


def _solid_points(values: Mapping[str, float]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x_key, y_key in (("10", "20"), ("11", "21"), ("12", "22"), ("13", "23")):
        if x_key in values and y_key in values:
            point = (float(values[x_key]), float(values[y_key]))
            if point not in points:
                points.append(point)
    return points


def _float_or_none(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _closed_flag(raw: str) -> bool:
    try:
        return bool(int(float(raw)) & 1)
    except ValueError:
        return False


def _record_id(record: str) -> str | None:
    match = re.match(r"\s*#(\d+)\s*=", record)
    return f"#{match.group(1)}" if match else None


def _sxf_entity_type(record: str) -> str | None:
    match = re.match(r"\s*(?:#\d+\s*=\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", record)
    return match.group(1).upper() if match else None


def _sxf_entity_item(record: str) -> dict[str, Any] | None:
    ident = _record_id(record)
    entity_type = _sxf_entity_type(record)
    if ident is None or entity_type is None:
        return None
    refs = [ref for ref in _record_refs(record) if ref != ident]
    item: dict[str, Any] = {
        "id": ident,
        "type": entity_type,
        "refs": refs,
        "labels": _quoted_strings(record),
        "numbers": _numbers_without_record_refs(record),
        "coordinates": [[x, y] for x, y in _coordinate_pairs(record)],
        "raw": record,
        "handled": False,
    }
    return item


def _mark_sxf_entity(entity: dict[str, Any] | None, mapped_as: str) -> None:
    if entity is None:
        return
    entity["handled"] = True
    mapped = entity.setdefault("mapped_as", [])
    if isinstance(mapped, list):
        if mapped_as not in mapped:
            mapped.append(mapped_as)
    else:
        entity["mapped_as"] = [mapped, mapped_as]


def _update_sxf_entity_style(entity: dict[str, Any] | None, *, layer: str | None = None, linetype: str | None = None) -> None:
    if entity is None:
        return
    if layer:
        entity["layer"] = layer
    if linetype:
        entity["linetype"] = linetype


def _finalize_sxf_entity_coverage(doc: CadDocument) -> None:
    entities = [entity for entity in _as_list(doc.get("entities", [])) if isinstance(entity, Mapping)]
    by_type: dict[str, int] = {}
    mapped_as: dict[str, int] = {}
    unhandled_types: dict[str, int] = {}
    handled = 0
    for entity in entities:
        etype = str(entity.get("type", "UNKNOWN"))
        by_type[etype] = by_type.get(etype, 0) + 1
        if bool(entity.get("handled", False)):
            handled += 1
        else:
            unhandled_types[etype] = unhandled_types.get(etype, 0) + 1
        mapped = entity.get("mapped_as", [])
        for item in mapped if isinstance(mapped, list) else [mapped]:
            key = str(item)
            mapped_as[key] = mapped_as.get(key, 0) + 1
    doc["sxf_entity_coverage"] = {
        "total": len(entities),
        "handled": handled,
        "unhandled": len(entities) - handled,
        "by_type": dict(sorted(by_type.items())),
        "mapped_as": dict(sorted(mapped_as.items())),
        "unhandled_types": dict(sorted(unhandled_types.items())),
        "preservation": "all_step_records_with_id",
    }


def _record_refs(record: str) -> list[str]:
    return [f"#{value}" for value in re.findall(r"#(\d+)", record)]


def _record_points(record: str, points: Mapping[str, tuple[float, float]]) -> list[tuple[float, float]]:
    return [points[ref] for ref in _record_refs(record) if ref in points]


def _detect_record_layer(record: str, layer_names: Mapping[str, str]) -> str | None:
    for ref in _record_refs(record):
        if ref in layer_names:
            return layer_names[ref]
    known = set(layer_names.values())
    for label in _quoted_strings(record):
        if label in known:
            return label
    return None


def _detect_record_linetype(record: str, linetype_names: Mapping[str, str]) -> str | None:
    for ref in _record_refs(record):
        if ref in linetype_names:
            return linetype_names[ref]
    known = set(linetype_names.values())
    for label in _quoted_strings(record):
        if label in known:
            return label
    return None


def _record_label(record: str) -> str | None:
    labels = [label for label in _quoted_strings(record) if label.strip()]
    return labels[-1] if labels else None


def _quoted_strings(record: str) -> list[str]:
    return [value.replace("''", "'") for value in re.findall(r"'((?:[^']|'')*)'", record)]


def _numbers(record: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", record)]


def _numbers_without_record_refs(record: str) -> list[float]:
    return _numbers(re.sub(r"#\d+", "", record))


def _coordinate_pairs(record: str) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in re.findall(
            r"\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
            record,
        )
    ]


def _last_coordinate_pair(record: str) -> tuple[float, float] | None:
    pairs = _coordinate_pairs(record)
    if not pairs:
        return None
    return pairs[-1]


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping_layer(value: Mapping[str, Any]) -> str | None:
    raw = value.get("layer", value.get("layer_name"))
    return str(raw) if raw not in (None, "") else None


def _xy_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            return None
        try:
            return float(value["x"]), float(value["y"])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


__all__ = [
    "CadDocument",
    "CadImportError",
    "Line2D",
    "_dwg_converter_command",
    "add_document_curve",
    "add_document_dimension_style",
    "add_document_hatch",
    "add_document_linetype",
    "arc_points",
    "arc_segments",
    "cad_document_line_tuples",
    "curve_points_from_mapping",
    "discover_dwg_converter",
    "document_from_mapping",
    "dwg_converter_requirement_message",
    "dxf_color_name",
    "ellipse_points",
    "empty_cad_document",
    "export_sxf_document",
    "export_sxf_file",
    "lines_from_mapping",
    "normalize_hatch_rings",
    "normalize_linetype_pattern",
    "parse_cad_document",
    "parse_cad_file",
    "parse_cad_file_document",
    "parse_cad_lines",
    "parse_dxf_document",
    "parse_dxf_lines",
    "parse_dwg_file",
    "parse_dwg_file_document",
    "parse_gf1_document",
    "parse_gf1_document_bytes",
    "parse_gf1_lines",
    "parse_gf1_lines_bytes",
    "parse_plain_lines",
    "parse_sxf_document",
    "parse_sxf_lines",
    "split_sxf_step_records",
    "validate_dwg_converter_link",
    "validate_sxf_roundtrip",
    "segments_from_points",
]
