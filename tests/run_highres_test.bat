@echo off
setlocal
set "VECTOR_TEST_REUSE="
cd /d "%~dp0.."
if not exist "python\python.exe" (
  echo Portable Python was not found at python\python.exe
  exit /b 2
)
"python\python.exe" "tests\highres_smoke.py"
exit /b %errorlevel%
