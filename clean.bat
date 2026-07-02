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

if not defined PYTHON_EXE goto NO_PYTHON

%PYTHON_EXE% -B -c "import PIL, numpy, vtracer" >nul 2>nul
if errorlevel 1 goto NO_DEPS

echo.
echo ==========================================
echo   AI Logo Vector Cleanroom
echo ==========================================
echo.

%PYTHON_EXE% -B "%~dp0vector_cleanroom.py" %*
echo.
pause
exit /b %ERRORLEVEL%

:NO_PYTHON
echo Python runtime was not found.
echo.
echo Please install Python 3, or ask for a packaged portable/exe version.
echo After installing Python, run install_deps.bat once.
echo.
pause
exit /b 1

:NO_DEPS
echo Missing Python packages: pillow / numpy / vtracer.
echo.
echo Please run install_deps.bat once, then run this file again.
echo.
pause
exit /b 1
