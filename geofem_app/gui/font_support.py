"""GUI font discovery and fallback helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PREFERRED_GUI_FONTS: tuple[str, ...] = (
    "Yu Gothic UI",
    "Yu Gothic",
    "Meiryo UI",
    "Meiryo",
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "MS Gothic",
    "Segoe UI",
)

KNOWN_JAPANESE_FONT_FILES: tuple[Path, ...] = (
    Path("C:/Windows/Fonts/YuGothR.ttc"),
    Path("C:/Windows/Fonts/YuGothM.ttc"),
    Path("C:/Windows/Fonts/meiryo.ttc"),
    Path("C:/Windows/Fonts/NotoSansJP-VF.ttf"),
    Path("C:/Windows/Fonts/msgothic.ttc"),
)


def register_known_japanese_fonts(font_database: Any) -> list[str]:
    """Register known Windows Japanese fonts when Qt did not enumerate them."""

    loaded: list[str] = []
    for path in KNOWN_JAPANESE_FONT_FILES:
        if not path.exists():
            continue
        font_id = int(font_database.addApplicationFont(str(path)))
        if font_id < 0:
            continue
        families = [str(name) for name in font_database.applicationFontFamilies(font_id)]
        loaded.extend(name for name in families if name)
    return loaded


def preferred_gui_font_inventory(font_database: Any) -> dict[str, Any]:
    """Return preferred GUI font availability, registering known local fonts first."""

    before = set(str(name) for name in font_database.families())
    loaded = register_known_japanese_fonts(font_database)
    families = set(str(name) for name in font_database.families())
    families.update(loaded)
    available = [font for font in PREFERRED_GUI_FONTS if font in families]
    return {
        "preferred": list(PREFERRED_GUI_FONTS),
        "available_preferred": available,
        "loaded_families": sorted(set(loaded)),
        "registered_known_font_count": len(set(loaded) - before),
    }


def apply_preferred_gui_font(app_obj: Any, font_database: Any, font_class: Any, point_size: int) -> dict[str, Any]:
    """Apply the first available preferred GUI font and annotate the QApplication."""

    inventory = preferred_gui_font_inventory(font_database)
    available = list(inventory["available_preferred"])
    if available:
        family = available[0]
        app_obj.setFont(font_class(family, point_size))
        app_obj.setProperty("geofemGuiFontFamily", family)
        app_obj.setProperty("geofemGuiFontCandidatesMissing", False)
        app_obj.setProperty("geofemGuiLoadedFonts", ",".join(inventory["loaded_families"]))
        return {**inventory, "selected": family, "status": "available"}
    family = str(app_obj.font().family())
    app_obj.setProperty("geofemGuiFontFamily", family)
    app_obj.setProperty("geofemGuiFontCandidatesMissing", True)
    app_obj.setProperty("geofemGuiLoadedFonts", ",".join(inventory["loaded_families"]))
    return {**inventory, "selected": family, "status": "missing_preferred"}


__all__ = [
    "PREFERRED_GUI_FONTS",
    "apply_preferred_gui_font",
    "preferred_gui_font_inventory",
    "register_known_japanese_fonts",
]
