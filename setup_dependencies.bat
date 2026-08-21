@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "LOG=%CD%\setup_dependencies.log"

echo [GeoFEM] Dependency setup started. > "%LOG%"
echo [GeoFEM] Working directory: %CD% >> "%LOG%"

if not exist "%VENV_PY%" (
    echo [GeoFEM] Creating local Python environment: %VENV_DIR%
    set "PY_CMD="
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            set "PY_CMD=py -3.12"
        ) else (
            set "PY_CMD=py -3"
        )
    )
    if not defined PY_CMD (
        where python >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
    if not defined PY_CMD (
        echo [GeoFEM] Python was not found. Install Python 3.12 or newer and retry.
        echo [GeoFEM] Python was not found. >> "%LOG%"
        exit /b 1
    )
    echo [GeoFEM] Python command: !PY_CMD! >> "%LOG%"
    !PY_CMD! -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo [GeoFEM] Python 3.12 or newer is required.
        echo [GeoFEM] Python version check failed. >> "%LOG%"
        exit /b 1
    )
    !PY_CMD! -m venv "%VENV_DIR%" >> "%LOG%" 2>&1
    if errorlevel 1 exit /b 1
)

echo [GeoFEM] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip==26.1.2 >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

if exist "%CD%\wheelhouse" (
    echo [GeoFEM] Installing dependencies from local wheelhouse...
    "%VENV_PY%" -m pip install --no-index --find-links "%CD%\wheelhouse" -r requirements.txt >> "%LOG%" 2>&1
) else (
    echo [GeoFEM] Installing locked dependencies from requirements.txt...
    "%VENV_PY%" -m pip install -r requirements.txt >> "%LOG%" 2>&1
)
if errorlevel 1 exit /b 1

echo [GeoFEM] Checking GUI dependencies...
"%VENV_PY%" -c "import PySide6, numpy, scipy, numba, yaml; print('GeoFEM GUI dependencies are ready.')" >> "%LOG%" 2>&1
if errorlevel 1 exit /b 1

echo [GeoFEM] Dependency setup completed. >> "%LOG%"
exit /b 0
