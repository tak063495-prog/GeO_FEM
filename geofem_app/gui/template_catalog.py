"""Low-allocation metadata indexing for GUI input templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml
from yaml.events import (
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
)


@dataclass(frozen=True)
class InputTemplateMetadata:
    top_level_keys: frozenset[str]
    analysis_type: str = ""
    element_type: str = ""
    integration: str = ""
    integration_variants: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {key: {} for key in self.top_level_keys}
        if self.analysis_type:
            data["analysis"] = {"type": self.analysis_type}
        mesh = dict(data.get("mesh", {}))
        if self.element_type:
            mesh["element_type"] = self.element_type
        if self.integration:
            mesh["integration"] = self.integration
        if mesh:
            data["mesh"] = mesh
        if self.integration_variants:
            data["template_variants"] = {"integrations": list(self.integration_variants)}
        return data


@dataclass
class _ContainerFrame:
    kind: str
    path: tuple[str, ...]
    expecting_key: bool = True
    key: str = ""


_METADATA_CACHE: dict[tuple[str, int, int], InputTemplateMetadata] = {}


def _container_value_path(parent: _ContainerFrame) -> tuple[str, ...]:
    if parent.kind == "mapping":
        path = parent.path + (parent.key,)
        parent.expecting_key = True
        parent.key = ""
        return path
    return parent.path


def _iter_scalar_paths(events: Iterator[Any]) -> Iterator[tuple[tuple[str, ...], str, bool]]:
    stack: list[_ContainerFrame] = []
    for event in events:
        if isinstance(event, MappingStartEvent):
            path = _container_value_path(stack[-1]) if stack else ()
            stack.append(_ContainerFrame("mapping", path))
            continue
        if isinstance(event, SequenceStartEvent):
            path = _container_value_path(stack[-1]) if stack else ()
            stack.append(_ContainerFrame("sequence", path, expecting_key=False))
            continue
        if isinstance(event, (MappingEndEvent, SequenceEndEvent)):
            if stack:
                stack.pop()
            continue
        if not isinstance(event, ScalarEvent) or not stack:
            continue
        parent = stack[-1]
        value = str(event.value or "")
        if parent.kind == "mapping" and parent.expecting_key:
            parent.key = value
            parent.expecting_key = False
            yield parent.path + (value,), value, True
            continue
        if parent.kind == "mapping":
            path = parent.path + (parent.key,)
            parent.expecting_key = True
            parent.key = ""
        else:
            path = parent.path
        yield path, value, False


def read_input_template_metadata(path: str | Path) -> InputTemplateMetadata:
    """Read only catalog metadata, stopping before large mesh/result payloads."""

    source = Path(path)
    stat = source.stat()
    key = (str(source.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    top_level_keys: set[str] = set()
    values: dict[tuple[str, ...], str] = {}
    integrations: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for scalar_path, value, is_key in _iter_scalar_paths(iter(yaml.parse(handle))):
            if is_key and len(scalar_path) == 1:
                top_level_keys.add(scalar_path[0])
            if is_key:
                continue
            if scalar_path in {
                ("analysis", "type"),
                ("mesh", "element_type"),
                ("mesh", "type"),
                ("mesh", "integration"),
                ("mesh", "elements", "type"),
                ("mesh", "elements", "element_type"),
                ("mesh", "elements", "integration"),
            }:
                values[scalar_path] = value
            elif scalar_path == ("template_variants", "integrations"):
                integrations.append(value)
            analysis_type = values.get(("analysis", "type"), "")
            element_type = values.get(
                ("mesh", "element_type"),
                values.get(("mesh", "type"), values.get(("mesh", "elements", "type"), values.get(("mesh", "elements", "element_type"), ""))),
            )
            integration = values.get(("mesh", "integration"), values.get(("mesh", "elements", "integration"), ""))
            if analysis_type and element_type and integration:
                break

    metadata = InputTemplateMetadata(
        top_level_keys=frozenset(top_level_keys),
        analysis_type=values.get(("analysis", "type"), ""),
        element_type=values.get(
            ("mesh", "element_type"),
            values.get(("mesh", "type"), values.get(("mesh", "elements", "type"), values.get(("mesh", "elements", "element_type"), ""))),
        ),
        integration=values.get(("mesh", "integration"), values.get(("mesh", "elements", "integration"), "")),
        integration_variants=tuple(dict.fromkeys(value for value in integrations if value)),
    )
    # Keep one current stat-key per path so repeated project switches remain bounded.
    for stale_key in [item for item in _METADATA_CACHE if item[0] == key[0] and item != key]:
        _METADATA_CACHE.pop(stale_key, None)
    _METADATA_CACHE[key] = metadata
    return metadata


def clear_input_template_metadata_cache() -> None:
    _METADATA_CACHE.clear()


__all__ = ["InputTemplateMetadata", "clear_input_template_metadata_cache", "read_input_template_metadata"]
