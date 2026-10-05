@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="

if not defined AVC_VENV_DIR (
    if not defined LOCALAPPDATA (
        echo [ERROR] LOCALAPPDATA is missing. Set AVC_VENV_DIR to an absolute path.
        if not defined AVC_NO_PAUSE pause
        exit /b 2
    )
    set "AVC_VENV_DIR=%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v0.6.0-alpha"
)
set "AVC_PYTHON=%AVC_VENV_DIR%\Scripts\python.exe"

if not exist "%AVC_PYTHON%" (
    echo [ERROR] The Designer Preview external Python environment is missing:
    echo   "%AVC_VENV_DIR%"
    echo Run setup_windows.bat first.
    if not defined AVC_NO_PAUSE pause
    exit /b 2
)

"%AVC_PYTHON%" -E -s -B "%~dp0environment_preflight.py" --full --strict-versions
if errorlevel 1 (
    echo [ERROR] Environment preflight failed. Run setup_windows.bat to repair it.
    if not defined AVC_NO_PAUSE pause
    exit /b 2
)

echo.
echo ==========================================
echo   AI Vector Cleanroom ^(Open Source v0.6.0-alpha^)
echo ==========================================
echo.
"%AVC_PYTHON%" -E -s -B "%~dp0vector_cleanroom.py" %*
set "AVC_EXIT_CODE=%ERRORLEVEL%"
echo.
if not defined AVC_NO_PAUSE pause
exit /b %AVC_EXIT_CODE%
