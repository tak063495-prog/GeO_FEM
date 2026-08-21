@echo off
setlocal
set "RUNNER=%~dp0dist\sustainability_2024_case2_strict_fos_005_20260725\run_case2_strict_fos_005.bat"

if not exist "%RUNNER%" (
  echo Case2 strict FOS runner was not found:
  echo %RUNNER%
  pause
  exit /b 1
)

call "%RUNNER%" %*
exit /b %ERRORLEVEL%
