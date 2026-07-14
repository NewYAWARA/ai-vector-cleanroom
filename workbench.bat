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
    echo Python runtime was not found. Run install_deps.bat first.
    pause
    exit /b 1
)

%PYTHON_EXE% -B "%~dp0workbench.py"
pause
