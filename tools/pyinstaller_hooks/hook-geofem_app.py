"""Keep Python sources beside the frozen archive for Numba cache locators.

Numba @njit(cache=True) checks the original source path at import time.
PYZ-only collection leaves those paths absent and prevents CLI/GUI startup.
Source-only loading gives functions absolute source filenames that Numba can locate.
pyz+py is insufficient: code loaded from PYZ retains relative co_filename paths.
"""
from pathlib import Path

module_collection_mode = {'geofem_app': 'py'}

# The startup doctor verifies these distribution helpers beside the package.
_root = Path(__file__).resolve().parents[2]
datas = [(str(_root / name), '.') for name in
         ('run_gui.bat', 'build_gui_exe.bat', 'requirements.txt')]
