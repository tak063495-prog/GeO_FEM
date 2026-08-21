@echo off
setlocal
set "STOPPER=%~dp0dist\sustainability_2024_case1-4_auto_srm_speed_guarded_20260612\stop_case1-4_srm.bat"

if not exist "%STOPPER%" (
  echo Case1-4 SRM stop command was not found:
  echo %STOPPER%
  pause
  exit /b 1
)

call "%STOPPER%" %*
exit /b %ERRORLEVEL%
