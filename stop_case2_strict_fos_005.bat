@echo off
setlocal
set "STOPPER=%~dp0dist\sustainability_2024_case2_strict_fos_005_20260725\stop_case2_strict_fos_005.bat"

if not exist "%STOPPER%" (
  echo Case2 strict FOS stop command was not found:
  echo %STOPPER%
  pause
  exit /b 1
)

call "%STOPPER%"
exit /b %ERRORLEVEL%
