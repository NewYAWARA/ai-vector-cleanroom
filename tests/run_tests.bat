@echo off
setlocal
set "VECTOR_TEST_REUSE="
cd /d "%~dp0.."
rem Prefer a portable interpreter if present, otherwise use system Python.
set "PYEXE=python"
if exist "python\python.exe" set "PYEXE=python\python.exe"
"%PYEXE%" -m unittest discover -s tests -p "test_*.py" -v
exit /b %errorlevel%
