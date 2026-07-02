@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHON_EXE="

if exist "%~dp0python\python.exe" set "PYTHON_EXE=%~dp0python\python.exe"

if not defined PYTHON_EXE (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=py -3"
)

if not defined PYTHON_EXE (
    echo Python runtime was not found.
    echo Please install Python 3 first, then run this file again.
    echo.
    pause
    exit /b 1
)

echo Installing required packages...
%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install -r "%~dp0requirements.txt"
echo.
echo Installing optional preview packages (safe to ignore if they fail)...
%PYTHON_EXE% -m pip install -r "%~dp0requirements-preview.txt"

echo.
echo Done. You can now run clean.bat.
echo.
pause
