"""Open seepage-result interchange helpers for GeoFEAS-style workflows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


EXTERNAL_SEEPAGE_PRODUCT_PROFILES: dict[str, dict[str, Any]] = {
    "generic_csv": {"delimiters": [",", "\t", ";"], "versions": ["generic"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
    "GeoFEAS": {"delimiters": [",", "\t"], "versions": ["auto"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
    "UC-1": {"delimiters": [",", "\t"], "versions": ["auto"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
    "VGFlow": {"delimiters": [",", "\t"], "versions": ["auto"], "required_fields": ["node_id"], "value_fields": ["pore_pressure", "head", "water_level"]},
}

_SEEPAGE_HEADER_ALIASES = {
    "time": {"time", "t", "step_time", "elapsed_time", "時刻", "時間", "経過時間"},
    "step": {"step", "stage", "ステップ", "段階"},
    "node_id": {"node_id", "node", "node_no", "node_number", "節点", "節点番号", "節点no", "節点No", "No"},
    "x": {"x", "x座標", "座標x"},
    "y": {"y", "y座標", "座標y"},
    "pore_pressure": {"pore_pressure", "pressure", "p", "water_pressure", "水圧", "間隙水圧", "過剰間隙水圧"},
    "head": {"head", "water_head", "total_head", "全水頭", "水頭", "水位標高"},
    "water_level": {"water_level", "level", "水位"},
    "unit": {"unit", "単位"},
}


def import_external_seepage_results(path: str | Path, *, product: str = "auto") -> list[dict[str, Any]]:
    """Read GeoFEAS/UC-1/VGFlow style seepage CSV/TSV output into normalized rows."""

    source = Path(path)
    rows = _read_delimited_rows(source)
    normalized: list[dict[str, Any]] = []
    product_name = _guess_seepage_product(source, rows, product)
    for row in rows:
        out: dict[str, Any] = {"source_product": product_name, "source_file": str(source)}
        for key, value in row.items():
            field = _normalize_seepage_header(key)
            if not field or value in (None, ""):
                continue
            if field == "node_id":
                out[field] = str(value).strip()
            elif field in {"time", "x", "y", "pore_pressure", "head", "water_level"}:
                out[field] = float(str(value).replace(",", "").strip())
            else:
                out[field] = str(value).strip()
        if "node_id" in out and any(field in out for field in ("pore_pressure", "head", "water_level")):
            out.setdefault("time", 0.0)
            normalized.append(out)
    return normalized


def compare_external_seepage_results(
    actual_path: str | Path,
    reference_path: str | Path,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Compare normalized external seepage results by time and node."""

    actual_rows = import_external_seepage_results(actual_path)
    reference_rows = import_external_seepage_results(reference_path)
    fields = ("pore_pressure", "head", "water_level")
    actual = {_seepage_key(row): row for row in actual_rows}
    reference = {_seepage_key(row): row for row in reference_rows}
    missing = sorted(set(reference) - set(actual))
    extra = sorted(set(actual) - set(reference))
    comparisons: list[dict[str, Any]] = []
    failed = 0
    max_abs = 0.0
    max_rel = 0.0
    for key in sorted(set(actual) & set(reference)):
        for field in fields:
            if field not in actual[key] or field not in reference[key]:
                continue
            av = float(actual[key][field])
            rv = float(reference[key][field])
            abs_err = abs(av - rv)
            rel_err = abs_err / max(abs(rv), atol)
            ok = abs_err <= atol + rtol * abs(rv)
            failed += 0 if ok else 1
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            comparisons.append({"key": key, "field": field, "actual": av, "reference": rv, "abs_error": abs_err, "rel_error": rel_err, "ok": ok})
    return {
        "actual": str(actual_path),
        "reference": str(reference_path),
        "rtol": rtol,
        "atol": atol,
        "row_count": len(comparisons),
        "missing_keys": missing,
        "extra_keys": extra,
        "failed_count": failed + len(missing),
        "passed": failed == 0 and not missing,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "comparisons": comparisons,
    }


def write_external_seepage_results(rows: Iterable[Mapping[str, Any]], path: str | Path, *, product: str = "generic_csv") -> None:
    """Write normalized seepage rows in the open interchange CSV used for round-trip checks."""

    fields = ["time", "node_id", "pore_pressure", "head", "water_level", "source_product", "source_version"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {field: row.get(field, "") for field in fields}
            item["source_product"] = row.get("source_product", product)
            writer.writerow(item)


def compare_external_seepage_roundtrip(
    source_path: str | Path,
    exported_path: str | Path,
    *,
    product: str = "auto",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Normalize, export, and compare an external seepage file through the open interchange path."""

    rows = import_external_seepage_results(source_path, product=product)
    write_external_seepage_results(rows, exported_path, product=rows[0].get("source_product", "generic_csv") if rows else "generic_csv")
    return compare_external_seepage_results(source_path, exported_path, rtol=rtol, atol=atol)


def diagnose_external_seepage_version(path: str | Path, *, product: str = "auto") -> dict[str, Any]:
    """Report detected product/version traits and unmapped columns for a seepage result file."""

    source = Path(path)
    raw_rows = _read_delimited_rows(source)
    product_name = _guess_seepage_product(source, raw_rows, product)
    headers = list(raw_rows[0].keys() if raw_rows else [])
    mapped = {header: _normalize_seepage_header(header) for header in headers}
    unsupported = [header for header, normalized in mapped.items() if not normalized]
    normalized_rows = import_external_seepage_results(path, product=product_name)
    version = _detect_external_version(source, raw_rows)
    profile = EXTERNAL_SEEPAGE_PRODUCT_PROFILES.get(product_name, EXTERNAL_SEEPAGE_PRODUCT_PROFILES["generic_csv"])
    return {
        "path": str(source),
        "product": product_name,
        "version": version,
        "profile_versions": list(profile.get("versions", [])),
        "headers": headers,
        "mapped_headers": mapped,
        "unsupported_headers": unsupported,
        "normalized_count": len(normalized_rows),
        "profile": profile,
    }


def _read_delimited_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    return [dict(row) for row in csv.DictReader(text.splitlines(), dialect=dialect)]


def _normalize_seepage_header(header: Any) -> str:
    text = str(header or "").strip().replace(" ", "_")
    lower = text.lower()
    for normalized, aliases in _SEEPAGE_HEADER_ALIASES.items():
        if text in aliases or lower in {alias.lower() for alias in aliases}:
            return normalized
    compact = lower.replace("_", "").replace("-", "")
    for normalized, aliases in _SEEPAGE_HEADER_ALIASES.items():
        if compact in {alias.lower().replace("_", "").replace("-", "") for alias in aliases}:
            return normalized
    return ""


def _guess_seepage_product(path: Path, rows: list[dict[str, str]], product: str) -> str:
    if product and product != "auto":
        return product
    joined = " ".join([path.name, *list(rows[0].keys() if rows else {})]).lower()
    if "uc-1" in joined or "uc1" in joined:
        return "UC-1"
    if "geofeas" in joined or "geo feas" in joined:
        return "GeoFEAS"
    if "vgflow" in joined:
        return "VGFlow"
    return "generic_csv"


def _detect_external_version(path: Path, rows: list[dict[str, str]]) -> str:
    joined = " ".join([path.name, *list(rows[0].keys() if rows else {}), *list((rows[0].values() if rows else []))]).lower()
    for marker in ("ver.", "version", "v"):
        idx = joined.find(marker)
        if idx >= 0:
            tail = joined[idx + len(marker) :].strip(" _:-")
            token = tail.split()[0].strip(",;") if tail else ""
            if token:
                return token
    return "auto"


def _seepage_key(row: Mapping[str, Any]) -> str:
    return f"{float(row.get('time', 0.0) or 0.0):.12g}|{row.get('node_id', '')}"


__all__ = [
    "EXTERNAL_SEEPAGE_PRODUCT_PROFILES",
    "compare_external_seepage_results",
    "compare_external_seepage_roundtrip",
    "diagnose_external_seepage_version",
    "import_external_seepage_results",
    "write_external_seepage_results",
]
