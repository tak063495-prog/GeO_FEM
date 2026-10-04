"""Run PyInstaller with Windows system DLLs ahead of unrelated tool DLLs."""
import os
import sys

if sys.platform == "win32":
    os.environ["PATH"] = os.path.join(os.environ["SystemRoot"], "System32") + os.pathsep + os.environ.get("PATH", "")

from PyInstaller.__main__ import run

if __name__ == "__main__":
    run(sys.argv[1:])
