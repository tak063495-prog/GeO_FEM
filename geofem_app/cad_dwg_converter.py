"""DWG converter discovery and command construction."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
from typing import Sequence

DEFAULT_DWG_CONVERTER_CANDIDATES = ("dwg2dxf", "dwg2dxf.exe", "dwgread", "dwgread.exe")


def dwg_converter_command(converter: str | Sequence[str], source: Path, output: Path) -> str | list[str]:
    replacements = {"input": str(source), "output": str(output)}
    if isinstance(converter, str):
        if "{input}" in converter or "{output}" in converter:
            return converter.format(**replacements)
        return f'"{converter}" "{source}" "{output}"'
    parts = [str(part) for part in converter]
    if any("{input}" in part or "{output}" in part for part in parts):
        return [part.format(**replacements) for part in parts]
    return [*parts, str(source), str(output)]


def discover_dwg_converter() -> str | Sequence[str] | None:
    configured = os.environ.get("GEOFEM_DWG_CONVERTER")
    if configured:
        return configured
    if os.environ.get("GEOFEM_DWG_AUTODISCOVER", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    for candidate in dwg_converter_candidates():
        executable = candidate if Path(candidate).exists() else shutil.which(candidate)
        if executable:
            return f'"{executable}" "{{input}}" "{{output}}"'
    return None


def dwg_converter_candidates() -> list[str]:
    raw = os.environ.get("GEOFEM_DWG_CONVERTER_CANDIDATES")
    if not raw:
        return list(DEFAULT_DWG_CONVERTER_CANDIDATES)
    return [part.strip().strip('"') for part in re.split(r"[;\n]", raw) if part.strip()]


def dwg_converter_requirement_message() -> str:
    names = ", ".join(DEFAULT_DWG_CONVERTER_CANDIDATES)
    return (
        "DWG import requires GEOFEM_DWG_CONVERTER or an auto-detected converter on PATH "
        f"({names}). Native DWG binary decoding is not implemented; convert through DXF/SXF."
    )


__all__ = [
    "DEFAULT_DWG_CONVERTER_CANDIDATES",
    "discover_dwg_converter",
    "dwg_converter_candidates",
    "dwg_converter_command",
    "dwg_converter_requirement_message",
]
