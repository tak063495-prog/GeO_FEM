@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "BUILD_VENV=%CD%\.build_exe_venv"
set "BUILD_PY=%BUILD_VENV%\Scripts\python.exe"
set "PYINSTALLER_HOOKS=%CD%\tools\pyinstaller_hooks"

if not exist "%BUILD_PY%" (
    py -3.12 -m venv "%BUILD_VENV%"
    if errorlevel 1 exit /b 1
)

"%BUILD_PY%" -m pip install --upgrade pip==26.1.2 wheel==0.47.0 setuptools==82.0.1
if errorlevel 1 exit /b 1
"%BUILD_PY%" -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 exit /b 1

set "PATH=%BUILD_VENV%\Library\bin;%PATH%"

"%BUILD_PY%" -m PyInstaller --noconfirm --clean --distpath build\pyinstaller_dist --workpath build\pyinstaller_work --specpath build\pyinstaller_spec --additional-hooks-dir "%PYINSTALLER_HOOKS%" --name GeoFEM-GUI --windowed --paths "%CD%" --collect-all numba --collect-all llvmlite geofem_gui.py
if errorlevel 1 exit /b 1
"%BUILD_PY%" -m PyInstaller --noconfirm --clean --distpath build\pyinstaller_dist --workpath build\pyinstaller_work --specpath build\pyinstaller_spec --additional-hooks-dir "%PYINSTALLER_HOOKS%" --name GeoFEM-CLI --console --paths "%CD%" --collect-all numba --collect-all llvmlite geofem_cli.py
if errorlevel 1 exit /b 1

copy /Y build\pyinstaller_dist\GeoFEM-CLI\GeoFEM-CLI.exe build\pyinstaller_dist\GeoFEM-GUI\GeoFEM-CLI.exe >nul
echo Built: build\pyinstaller_dist\GeoFEM-GUI
