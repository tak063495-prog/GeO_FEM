"""Size-guarded file readers for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_TEXT_PREVIEW_BYTES = 512 * 1024
DEFAULT_INPUT_PARSE_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class TextFilePreview:
    path: str
    text: str
    size_bytes: int
    truncated: bool
    preview_bytes: int


def preview_text_file(path: str | Path, *, max_bytes: int = DEFAULT_TEXT_PREVIEW_BYTES, encoding: str = "utf-8") -> TextFilePreview:
    source = Path(path)
    size = source.stat().st_size
    limit = max(1, int(max_bytes))
    with source.open("rb") as f:
        data = f.read(min(size, limit))
    text = data.decode(encoding, errors="replace")
    truncated = size > limit
    if truncated:
        text += (
            "\n\n--- GUI preview truncated ---\n"
            f"path: {source}\n"
            f"size_bytes: {size}\n"
            f"preview_bytes: {limit}\n"
            "Open the file externally or use a filtered result table for the full content.\n"
        )
    return TextFilePreview(path=str(source), text=text, size_bytes=size, truncated=truncated, preview_bytes=min(size, limit))


def read_mapping_file_guarded(path: str | Path, *, max_bytes: int = DEFAULT_INPUT_PARSE_BYTES) -> tuple[dict[str, Any], TextFilePreview]:
    source = Path(path)
    size = source.stat().st_size
    if size > max_bytes:
        preview = preview_text_file(source, max_bytes=min(max_bytes, DEFAULT_TEXT_PREVIEW_BYTES))
        raise ValueError(f"input file is too large for synchronous GUI parsing: {source} ({size} bytes, limit {max_bytes} bytes)")
    text = source.read_text(encoding="utf-8")
    data = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError("input root must be a mapping")
    return dict(data), TextFilePreview(path=str(source), text=text, size_bytes=size, truncated=False, preview_bytes=size)


def read_json_file_guarded(path: str | Path, *, max_bytes: int = DEFAULT_INPUT_PARSE_BYTES) -> Any:
    source = Path(path)
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"JSON file is too large for synchronous GUI parsing: {source} ({size} bytes, limit {max_bytes} bytes)")
    return json.loads(source.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_INPUT_PARSE_BYTES",
    "DEFAULT_TEXT_PREVIEW_BYTES",
    "TextFilePreview",
    "preview_text_file",
    "read_json_file_guarded",
    "read_mapping_file_guarded",
]
