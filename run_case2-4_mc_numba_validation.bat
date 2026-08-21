@echo off
setlocal
set "RUNNER=%~dp0tools\run_case2-4_mc_numba_validation.ps1"

if not exist "%RUNNER%" (
  echo Validation runner was not found:
  echo %RUNNER%
  pause
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%RUNNER%" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Case2 and Case4 validation finished.
) else (
  echo Validation finished with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
