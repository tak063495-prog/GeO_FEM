@echo off
setlocal
set "POINTER=%~dp0last_case2-4_mc_numba_validation.txt"

if not exist "%POINTER%" (
  echo No Case2-Case4 validation run was found.
  pause
  exit /b 1
)

set /p RUN_DIR=<"%POINTER%"
if not defined RUN_DIR (
  echo The validation run pointer is empty.
  pause
  exit /b 1
)

if exist "%RUN_DIR%\case2" (
  type nul > "%RUN_DIR%\case2\cancel.request"
)
if exist "%RUN_DIR%\case4_runner" (
  type nul > "%RUN_DIR%\case4_runner\cancel.request"
)

echo Cancellation was requested:
echo %RUN_DIR%
pause
