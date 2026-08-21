@echo off
setlocal
set "RESUME_RUNNER=%~dp0dist\sustainability_2024_case1-4_auto_srm_speed_guarded_20260612\resume_last_case1-4_srm.bat"

if not exist "%RESUME_RUNNER%" (
  echo Case1-4 SRM resume runner was not found:
  echo %RESUME_RUNNER%
  pause
  exit /b 1
)

call "%RESUME_RUNNER%"
exit /b %ERRORLEVEL%
