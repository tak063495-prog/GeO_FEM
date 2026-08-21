"""Desktop layout profile helpers for the PySide GUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DesktopLayoutProfile:
    available_width: int
    available_height: int
    window_width: int
    window_height: int
    horizontal_split_sizes: tuple[int, int, int]
    vertical_split_sizes: tuple[int, int]
    minimum_window_size: tuple[int, int]
    tree_min_width: int
    center_min_width: int
    panel_min_width: int
    model_view_min_size: tuple[int, int]
    large_desktop: bool


def resolve_desktop_layout_profile(available_width: int, available_height: int) -> DesktopLayoutProfile:
    """Resolve a screen-aware GUI layout profile.

    The 2560x1440 target keeps the window below the available desktop while
    giving the central model/Post area most of the extra width.
    """

    width = max(1024, int(available_width))
    height = max(700, int(available_height))
    large = width >= 2400 and height >= 1300
    narrow = width < 1200

    if large:
        window_width = min(width - 120, int(width * 0.90))
        window_height = min(height - 90, int(height * 0.90))
        left = max(300, int(window_width * 0.14))
        right = max(440, int(window_width * 0.20))
        bottom = max(260, int(window_height * 0.20))
    elif narrow:
        window_width = width
        window_height = height
        left = max(196, int(window_width * 0.20))
        right = max(260, int(window_width * 0.26))
        bottom = 42
    else:
        window_width = min(width - 40, max(1280, int(width * 0.92)))
        window_height = min(height - 50, max(760, int(height * 0.88)))
        left = max(264, int(window_width * 0.19))
        right = max(360, int(window_width * 0.25))
        bottom = max(190, int(window_height * 0.22))

    window_width = max(1024, window_width)
    window_height = max(700, window_height)
    tree_min_width = 196 if narrow else (248 if not large else 260)
    center_min_width = 390 if narrow else (520 if not large else 900)
    panel_min_width = 260 if narrow else (330 if not large else 380)
    left = min(left, max(tree_min_width, int(window_width * 0.22)))
    right = min(right, max(panel_min_width, int(window_width * 0.28)))
    center = max(center_min_width, window_width - left - right)
    if left + center + right != window_width:
        center = window_width - left - right

    bottom = min(bottom, max(180, int(window_height * 0.32)))
    top = max(420 if narrow else 480, window_height - bottom)
    if top + bottom != window_height:
        top = window_height - bottom

    return DesktopLayoutProfile(
        available_width=width,
        available_height=height,
        window_width=window_width,
        window_height=window_height,
        horizontal_split_sizes=(left, center, right),
        vertical_split_sizes=(top, bottom),
        minimum_window_size=(1024, 700) if narrow else (1280, 720),
        tree_min_width=tree_min_width,
        center_min_width=center_min_width,
        panel_min_width=panel_min_width,
        model_view_min_size=(360, 260) if narrow else ((480, 320) if not large else (900, 620)),
        large_desktop=large,
    )


__all__ = ["DesktopLayoutProfile", "resolve_desktop_layout_profile"]
