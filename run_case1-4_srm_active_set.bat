@echo off
setlocal
set "RUNNER=%~dp0dist\sustainability_2024_case1-4_auto_srm_speed_guarded_20260612\run_case1-4_srm_active_set.bat"

if not exist "%RUNNER%" (
  echo Case1-4 SRM active-set runner was not found:
  echo %RUNNER%
  pause
  exit /b 1
)

call "%RUNNER%" %*
exit /b %ERRORLEVEL%
