"""Writable project-location helpers for the desktop GUI."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


_INSTALL_LOCATION_MARKERS = {
    "program files",
    "program files (x86)",
    "windowsapps",
}


def is_writable_directory(path: str | Path) -> bool:
    """Return whether *path* can host GeoFEM's project artifacts."""

    candidate = Path(path).expanduser()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".geofem-write-", dir=candidate, delete=True):
            pass
    except (OSError, PermissionError):
        return False
    return True


def is_install_location(path: str | Path) -> bool:
    """Identify locations that should never be used as an implicit project root."""

    parts = {part.casefold() for part in Path(path).expanduser().parts}
    return bool(parts.intersection(_INSTALL_LOCATION_MARKERS))


def resolve_initial_project_root(
    cwd: str | Path | None = None,
    *,
    home: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path:
    """Choose a writable, user-owned initial project root.

    An explicit ``GEOFEM_PROJECT_ROOT`` wins. A normal writable working
    directory is retained for source and portable launches, while protected
    installation folders fall back to the user's Documents directory.
    """

    env = os.environ if environ is None else environ
    user_home = Path(home).expanduser() if home is not None else Path.home()
    candidates: list[Path] = []
    configured = str(env.get("GEOFEM_PROJECT_ROOT", "")).strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    working = Path.cwd() if cwd is None else Path(cwd).expanduser()
    if not is_install_location(working):
        candidates.append(working)
    candidates.extend((user_home / "Documents" / "GeoFEM", user_home / "GeoFEM"))
    for candidate in candidates:
        if is_writable_directory(candidate):
            return candidate.resolve()
    return user_home.resolve()


__all__ = ["is_install_location", "is_writable_directory", "resolve_initial_project_root"]
