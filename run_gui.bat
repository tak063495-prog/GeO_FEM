@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "GUI_LOG=%CD%\gui_start.log"

echo [GeoFEM] GUI launcher started. > "%GUI_LOG%"
echo [GeoFEM] Working directory: %CD% >> "%GUI_LOG%"

if not exist "%VENV_PY%" (
    call "%~dp0setup_dependencies.bat"
    if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import PySide6, numpy, scipy, numba, yaml" >> "%GUI_LOG%" 2>&1
if errorlevel 1 (
    call "%~dp0setup_dependencies.bat"
    if errorlevel 1 goto :error
)

"%VENV_PY%" -m geofem_app.cli gui >> "%GUI_LOG%" 2>&1
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [GeoFEM] GUI could not be started.
echo Check that Python 3.12 or newer is installed and that network access is available for the first dependency install.
echo Details are saved in:
echo   %GUI_LOG%
echo   %CD%\setup_dependencies.log
pause
exit /b 1
