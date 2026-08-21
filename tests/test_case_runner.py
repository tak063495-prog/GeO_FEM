from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CaseRunnerTests(unittest.TestCase):
    def test_native_stderr_warning_does_not_fail_case(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_package = (
            project_root
            / "dist"
            / "sustainability_2024_case1-4_auto_srm_speed_guarded_20260612"
        )
        source_runner = source_package / "run_case1-4_srm.ps1"
        self.assertTrue(source_runner.is_file())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package = tmp_path / "package"
            package.mkdir()
            runner = package / source_runner.name
            shutil.copy2(source_runner, runner)
            shutil.copy2(
                source_package
                / "sustainability_2024_case1_quad4_sri_auto_srm_speed_guarded.yaml",
                package
                / "sustainability_2024_case1_quad4_sri_auto_srm_speed_guarded.yaml",
            )
            mock_cli = tmp_path / "mock_geofem_cli.cmd"
            mock_cli.write_text(
                "\r\n".join(
                    (
                        "@echo off",
                        "setlocal EnableExtensions",
                        ">&2 echo NumbaPerformanceWarning: benign mock warning",
                        'set "OUT="',
                        ":args",
                        'if "%~1"=="" goto done',
                        'if /I "%~1"=="--out" (',
                        '  set "OUT=%~2"',
                        "  shift",
                        ")",
                        "shift",
                        "goto args",
                        ":done",
                        'if not defined OUT exit /b 2',
                        'if not exist "%OUT%" mkdir "%OUT%"',
                        '>"%OUT%\\summary.json" echo {"stages":[]}',
                        "exit /b 0",
                    )
                )
                + "\r\n",
                encoding="ascii",
            )
            output_root = tmp_path / "run"
            env = os.environ.copy()
            env["GEOFEM_CLI"] = str(mock_cli)
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(runner),
                    "-Cases",
                    "1",
                    "-OutputRoot",
                    str(output_root),
                    "-Workers",
                    "1",
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            summary = json.loads(
                (output_root / "case1-4_run_summary.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["cases"][0]["status"], "completed")
            console_log = (output_root / "case1" / "console.log").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("NumbaPerformanceWarning", console_log)
            self.assertNotIn("NativeCommandError", console_log)


if __name__ == "__main__":
    unittest.main()
