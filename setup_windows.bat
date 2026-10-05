@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INPUT=1"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Windows Python Launcher ^(py.exe^) was not found.
    echo Install 64-bit CPython 3.12 from python.org, including Python Launcher.
    if not defined AVC_NO_PAUSE pause
    exit /b 2
)

py -3.12 -E -s -c "import struct,sys; raise SystemExit(0 if sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 12) and struct.calcsize('P') * 8 == 64 else 1)"
if errorlevel 1 (
    echo [ERROR] 64-bit CPython 3.12 is required.
    if not defined AVC_NO_PAUSE pause
    exit /b 2
)

py -3.12 -E -s -B "%~dp0setup_windows.py"
set "AVC_SETUP_EXIT=%ERRORLEVEL%"
echo.
if not defined AVC_NO_PAUSE pause
exit /b %AVC_SETUP_EXIT%
