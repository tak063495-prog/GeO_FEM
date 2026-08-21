from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyLockTests(unittest.TestCase):
    def test_runtime_and_build_requirements_share_an_exact_lock(self) -> None:
        runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        build = (ROOT / "requirements-build.txt").read_text(encoding="utf-8")
        lock_lines = [
            line.strip()
            for line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertIn("-c requirements-lock.txt", runtime)
        self.assertIn("-c requirements-lock.txt", build)
        self.assertTrue(all("==" in line for line in lock_lines))
        locked_names = {line.split("==", 1)[0].lower().replace("_", "-") for line in lock_lines}
        for name in ("numpy", "scipy", "numba", "tbb", "pyyaml", "pyside6", "psutil", "threadpoolctl", "pyinstaller"):
            self.assertIn(name, locked_names)

    def test_windows_launchers_use_python_312_and_the_locked_build_requirements(self) -> None:
        setup = (ROOT / "setup_dependencies.bat").read_text(encoding="utf-8")
        build = (ROOT / "build_gui_exe.bat").read_text(encoding="utf-8")
        self.assertIn("py -3.12", setup)
        self.assertIn("sys.version_info >= (3, 12)", setup)
        self.assertIn("-r requirements.txt -r requirements-build.txt", build)


if __name__ == "__main__":
    unittest.main()
