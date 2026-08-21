"""Boundary curve package exports for the VGFlow 2D public substitute."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .fem2d_utils import _ensure_list
from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


CurveReader = Callable[[str | Path], list[dict[str, float]]]


def write_vgflow_curve_package(out: Path, seepage: Mapping[str, Any], curve_reader: Callable[..., list[dict[str, float]]]) -> dict[str, str]:
    paths = {
        "curve_package_json": str(out / "vgflow_curve_package.json"),
        "curve_package_csv": str(out / "vgflow_curve_package.csv"),
        "curve_package_html": str(out / "vgflow_curve_package.html"),
    }
    curves = _collect_curves(seepage, curve_reader)
    manifest = {
        "schema": "geofem.vgflow2d.curve_package.public_substitute.v1",
        "profile": "Open VGFlow 2D boundary-curve package; not a proprietary commercial curve binary.",
        "commercial_curve_binary_equivalence": False,
        "features": [
            "open_curve_manifest",
            "csv_ascii_curve_interchange",
            "inline_curve_catalog",
            "boundary_curve_role_mapping",
        ],
        "supported_open_formats": ["csv", "txt", "fcd", "qcd", "ascii whitespace table"],
        "curve_count": len(curves),
        "curves": curves,
    }
    write_json_artifact(paths["curve_package_json"], manifest)
    _write_csv(Path(paths["curve_package_csv"]), curves)
    write_html_artifact(paths["curve_package_html"], _html(manifest))
    return paths


def _collect_curves(seepage: Mapping[str, Any], curve_reader: Callable[..., list[dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, index, spec in _all_boundary_specs(seepage):
        for key in _curve_keys(spec):
            source = _curve_file_from_spec(spec, key)
            pairs = _curve_pairs(spec, key, curve_reader)
            if not pairs:
                continue
            values = [value for _time, value in pairs]
            times = [time for time, _value in pairs]
            rows.append(
                {
                    "curve_id": f"{group}_{index}_{key}",
                    "boundary_group": group,
                    "boundary_index": index,
                    "value_key": key,
                    "source": str(source or "inline"),
                    "source_kind": "file" if source not in (None, "") else "inline",
                    "point_count": len(pairs),
                    "time_min": min(times),
                    "time_max": max(times),
                    "value_min": min(values),
                    "value_max": max(values),
                    "units": _unit_for_key(key, spec),
                    "role": _role_for_key(key),
                    "points": [{"time": time, "value": value} for time, value in pairs],
                }
            )
    return rows


def _all_boundary_specs(seepage: Mapping[str, Any]) -> list[tuple[str, int, Mapping[str, Any]]]:
    groups = (
        "known_head_bcs",
        "head_boundaries",
        "water_level_boundaries",
        "pressure_head_boundaries",
        "rainfall",
        "rainfall_boundaries",
        "flux_boundaries",
        "flow_boundaries",
        "point_sources",
        "seepage_faces",
    )
    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for group in groups:
        raw = seepage.get(group)
        if isinstance(raw, Mapping):
            specs = [raw]
        else:
            specs = [item for item in _ensure_list(raw) if isinstance(item, Mapping)]
        rows.extend((group, index, spec) for index, spec in enumerate(specs))
    return rows


def _curve_keys(spec: Mapping[str, Any]) -> list[str]:
    candidates = ("head", "water_level", "pressure_head", "rainfall", "flux", "q", "flow", "value")
    keys = [
        key
        for key in candidates
        if any(spec.get(field) is not None for field in (f"{key}_curve", f"{key}_curve_file", f"{key}_time_series_file", f"{key}_file"))
    ]
    if keys:
        return keys
    if not any(spec.get(field) is not None for field in ("curve", "time_series", "curve_file", "time_series_file")):
        return []
    kind = str(spec.get("type", spec.get("kind", ""))).lower()
    if kind == "rainfall" or "rainfall" in spec:
        return ["rainfall"]
    if kind == "flux" or any(key in spec for key in ("flux", "q", "flow")):
        return ["flux"]
    if "pressure_head" in spec:
        return ["pressure_head"]
    return ["head"]


def _curve_pairs(spec: Mapping[str, Any], key: str, curve_reader: Callable[..., list[dict[str, float]]]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    source = _curve_file_from_spec(spec, key)
    if source not in (None, ""):
        pairs.extend((float(row["time"]), float(row[key])) for row in curve_reader(source, value_field=key))
    inline = spec.get(f"{key}_curve", spec.get("curve", spec.get("time_series")))
    if inline is not None:
        pairs.extend(_inline_pairs(inline, key))
    return sorted(pairs)


def _curve_file_from_spec(spec: Mapping[str, Any], key: str) -> Any:
    for field in (f"{key}_curve_file", f"{key}_time_series_file", f"{key}_file", "curve_file", "time_series_file"):
        value = spec.get(field)
        if value not in (None, ""):
            return value
    if "file" in spec and any(marker in spec for marker in ("curve", "time_series", "curve_file", "time_series_file")):
        return spec["file"]
    return None


def _inline_pairs(curve: Any, key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in _ensure_list(curve):
        if isinstance(row, Mapping):
            value = row.get(key, row.get("value", row.get("head", row.get("water_level", row.get("rainfall", row.get("flux", 0.0))))))
            pairs.append((float(row.get("time", row.get("t", 0.0)) or 0.0), float(value or 0.0)))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            pairs.append((float(row[0]), float(row[1])))
    return pairs


def _unit_for_key(key: str, spec: Mapping[str, Any]) -> str:
    unit = spec.get("unit", spec.get(f"{key}_unit"))
    if unit:
        return str(unit)
    defaults = {
        "head": "m",
        "water_level": "m",
        "pressure_head": "m",
        "rainfall": "m/s",
        "flux": "m/s",
        "q": "m3/s",
        "flow": "m3/s",
        "value": "-",
    }
    return defaults.get(key, "-")


def _role_for_key(key: str) -> str:
    if key in {"head", "water_level", "pressure_head"}:
        return "head_boundary_time_series"
    if key == "rainfall":
        return "rainfall_boundary_time_series"
    if key in {"flux", "q", "flow"}:
        return "flux_boundary_time_series"
    return "generic_boundary_time_series"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["curve_id", "boundary_group", "boundary_index", "value_key", "source", "source_kind", "point_count", "time_min", "time_max", "value_min", "value_max", "units", "role"]
    write_dict_rows_csv(path, rows, fields)


def _html(manifest: Mapping[str, Any]) -> str:
    rows = [
        [
            row.get("curve_id", ""),
            row.get("value_key", ""),
            row.get("source", ""),
            row.get("point_count", ""),
            row.get("time_min", ""),
            row.get("time_max", ""),
            row.get("units", ""),
        ]
        for row in manifest.get("curves", [])
    ]
    return html_table_document(
        title="VGFlow 2D Curve Package",
        lead="商用曲線バイナリではなく、CSV/ASCII/インライン曲線を境界条件の役割付きで整理する公開代替パッケージです。",
        headers=["curve", "key", "source", "points", "time min", "time max", "units"],
        rows=rows,
    )
__all__ = ["write_vgflow_curve_package"]
