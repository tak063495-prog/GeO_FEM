from importlib.util import find_spec

from PyInstaller.utils.hooks import is_module_satisfies


def _module_exists(name):
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# Local override for PyInstaller 6.20's SciPy hook. Some SciPy wheels are
# versioned >= 1.13 but do not ship scipy.special._cdflib, so every optional
# extension is guarded by actual module availability.
hiddenimports = []

if _module_exists("scipy.special._ufuncs_cxx"):
    hiddenimports.append("scipy.special._ufuncs_cxx")

if is_module_satisfies("scipy >= 1.13.0") and _module_exists("scipy.special._cdflib"):
    hiddenimports.append("scipy.special._cdflib")

if is_module_satisfies("scipy >= 1.14.0") and _module_exists("scipy.special._special_ufuncs"):
    hiddenimports.append("scipy.special._special_ufuncs")
