"""Built-in load-combination templates for 2D design checks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping


LOAD_COMBINATION_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "river_seismic": [
        {"name": "river_service", "description": "river/levee service seed", "clause": "open-template:river:service", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "water": 1.0, "hydro": 1.0, "live": 1.0}},
        {"name": "river_l1_seismic", "description": "river/levee level-1 seismic seed", "clause": "open-template:river:l1-seismic", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "water": 1.0, "hydro": 1.0, "earthquake": 1.0, "seismic": 1.0}},
        {"name": "river_l2_seismic", "description": "river/levee level-2 seismic seed", "clause": "open-template:river:l2-seismic", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "water": 1.0, "hydro": 1.0, "earthquake": 1.0, "seismic": 1.0}},
    ],
    "road_bridge": [
        {"name": "road_service", "description": "road/bridge service seed", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "live": 1.0, "water": 1.0}},
        {"name": "road_seismic", "description": "road/bridge seismic seed", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "earthquake": 1.0, "seismic": 1.0, "water": 1.0}},
    ],
    "railway": [
        {"name": "rail_service", "description": "railway service seed", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "live": 1.0, "train": 1.0, "water": 1.0}},
        {"name": "rail_seismic", "description": "railway seismic seed", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "earthquake": 1.0, "seismic": 1.0, "water": 1.0}},
    ],
    "port": [
        {"name": "port_service", "description": "port/coastal service seed", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "wave": 1.0, "water": 1.0}},
        {"name": "port_seismic", "description": "port/coastal seismic seed", "factors_by_type": {"dead": 1.0, "self_weight": 1.0, "earthquake": 1.0, "seismic": 1.0, "water": 1.0}},
    ],
}

LOAD_COMBINATION_TEMPLATE_META: dict[str, dict[str, Any]] = {
    "river_seismic": {"revision": "open-2026.05", "coverage": "seed", "note": "Open seed template; verify project-specific official clauses before production use."},
    "road_bridge": {"revision": "open-2026.05", "coverage": "seed", "note": "Open seed template; verify project-specific official clauses before production use."},
    "railway": {"revision": "open-2026.05", "coverage": "seed", "note": "Open seed template; verify project-specific official clauses before production use."},
    "port": {"revision": "open-2026.05", "coverage": "seed", "note": "Open seed template; verify project-specific official clauses before production use."},
}

_STANDARD_ALIASES = {
    "river": "river_seismic",
    "river_earthquake": "river_seismic",
    "river_seismic": "river_seismic",
    "kasen": "river_seismic",
    "road": "road_bridge",
    "road_bridge": "road_bridge",
    "bridge": "road_bridge",
    "rail": "railway",
    "railway": "railway",
    "port": "port",
    "harbor": "port",
}


def load_combination_templates_for_config(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    standard = _standard_from_config(cfg)
    if not standard:
        return []
    cases = _case_map(cfg)
    rows: list[dict[str, Any]] = []
    for template in LOAD_COMBINATION_TEMPLATES.get(standard, []):
        factors = _factors_for_cases(template, cases)
        if not factors and not cases:
            factors = dict(template.get("factors_by_type", {}))
        row = {key: value for key, value in template.items() if key != "factors_by_type"}
        row["standard"] = standard
        row["source"] = "built_in_design_standard_seed"
        row["revision"] = LOAD_COMBINATION_TEMPLATE_META.get(standard, {}).get("revision", "")
        row["coverage"] = LOAD_COMBINATION_TEMPLATE_META.get(standard, {}).get("coverage", "seed")
        row["factors"] = factors
        rows.append(row)
    return rows


def configured_load_combinations(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = []
    for combo in _ensure_list(cfg.get("load_combinations", cfg.get("load_combination_table", []))):
        if isinstance(combo, Mapping):
            explicit.append(dict(combo))
    return explicit + load_combination_templates_for_config(cfg)


def load_combination_template_manifest(standard: str | None = None) -> list[dict[str, Any]]:
    standards = [_standard_alias(standard)] if standard else sorted(LOAD_COMBINATION_TEMPLATES)
    rows: list[dict[str, Any]] = []
    for key in standards:
        if key not in LOAD_COMBINATION_TEMPLATES:
            continue
        meta = LOAD_COMBINATION_TEMPLATE_META.get(key, {})
        for template in LOAD_COMBINATION_TEMPLATES[key]:
            rows.append(
                {
                    "standard": key,
                    "revision": meta.get("revision", ""),
                    "coverage": meta.get("coverage", "seed"),
                    "combination": template.get("name", ""),
                    "clause": template.get("clause", ""),
                    "description": template.get("description", ""),
                    "case_types": ",".join(sorted(str(name) for name in template.get("factors_by_type", {}))),
                    "note": meta.get("note", ""),
                }
            )
    return rows


def write_load_combination_template_manifest(path: str | Path, standard: str | None = None) -> None:
    rows = load_combination_template_manifest(standard)
    fields = ["standard", "revision", "coverage", "combination", "clause", "description", "case_types", "note"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_load_combination_template_revision(base_standard: str, new_standard: str) -> dict[str, Any]:
    base = {row["combination"]: row for row in load_combination_template_manifest(base_standard)}
    new = {row["combination"]: row for row in load_combination_template_manifest(new_standard)}
    return {
        "base_standard": _standard_alias(base_standard),
        "new_standard": _standard_alias(new_standard),
        "added": sorted(set(new) - set(base)),
        "removed": sorted(set(base) - set(new)),
        "changed": sorted(name for name in set(base) & set(new) if base[name] != new[name]),
    }


def _standard_from_config(cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("load_combination_standard", cfg.get("combination_standard", cfg.get("design_standard", "")))
    if isinstance(raw, Mapping):
        raw = raw.get("name", raw.get("standard", ""))
    key = str(raw or "").lower().strip().replace("-", "_").replace(" ", "_")
    return _standard_alias(key)


def _standard_alias(value: Any) -> str:
    key = str(value or "").lower().strip().replace("-", "_").replace(" ", "_")
    return _STANDARD_ALIASES.get(key, key if key in LOAD_COMBINATION_TEMPLATES else "")


def _case_map(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in _ensure_list(cfg.get("load_cases", [])):
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name", raw.get("case", ""))).strip()
        if name:
            out[name] = dict(raw)
    return out


def _factors_for_cases(template: Mapping[str, Any], cases: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    by_type = template.get("factors_by_type", {})
    if not isinstance(by_type, Mapping):
        return {}
    factors: dict[str, float] = {}
    for name, case in cases.items():
        aliases = _case_aliases(name, case)
        for alias in aliases:
            if alias in by_type:
                factors[name] = float(by_type[alias])
                break
    return factors


def _case_aliases(name: str, case: Mapping[str, Any]) -> list[str]:
    values = [name, str(case.get("type", "")), str(case.get("category", "")), str(case.get("kind", ""))]
    aliases: list[str] = []
    for value in values:
        key = value.lower().strip().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        aliases.append(key)
        if key in {"selfweight", "gravity"}:
            aliases.append("self_weight")
        if key in {"eq", "earth_quake"}:
            aliases.extend(["earthquake", "seismic"])
        if key in {"water_pressure", "pore_pressure", "hydraulic"}:
            aliases.extend(["water", "hydro"])
    return aliases


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


__all__ = [
    "LOAD_COMBINATION_TEMPLATE_META",
    "LOAD_COMBINATION_TEMPLATES",
    "compare_load_combination_template_revision",
    "configured_load_combinations",
    "load_combination_template_manifest",
    "load_combination_templates_for_config",
    "write_load_combination_template_manifest",
]
