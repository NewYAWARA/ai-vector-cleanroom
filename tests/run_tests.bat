@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0.."

set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="
set "VECTOR_TEST_REUSE="

if defined AVC_VENV_DIR (
  set "AVC_ACTIVE_VENV=%AVC_VENV_DIR%"
) else (
  if not defined LOCALAPPDATA (
    echo [ERROR] LOCALAPPDATA is missing and AVC_VENV_DIR is not set.
    exit /b 2
  )
  set "AVC_ACTIVE_VENV=%LOCALAPPDATA%\AI-Vector-Cleanroom\venvs\v3-designer-preview.4"
)

set "AVC_PYTHON=%AVC_ACTIVE_VENV%\Scripts\python.exe"
if not exist "%AVC_PYTHON%" (
  echo [ERROR] AI Vector Cleanroom external venv was not found:
  echo         "%AVC_PYTHON%"
  echo Run setup_windows.bat from the project root first.
  exit /b 2
)

"%AVC_PYTHON%" -E -s -B -m unittest discover -s tests -p "test_*.py" -v
exit /b %errorlevel%
