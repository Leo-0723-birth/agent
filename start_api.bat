@echo off
chcp 65001 >nul
title FastAPI Service - Risk Early Warning System
cd /d "%~dp0"

set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

if "%API_PORT%"=="" set API_PORT=8000
if "%API_HOST%"=="" set API_HOST=0.0.0.0

echo ============================================================
echo   Regulatory Inquiry Risk Early Warning System (FastAPI)
echo   Access URLs after startup:
echo     Local:   http://localhost:%API_PORT%
echo     Docs:    http://localhost:%API_PORT%/docs
echo   Close this window to stop the service.
echo ============================================================
echo.

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

%PYTHON_EXE% -c "import uvicorn" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python or uvicorn is unavailable.
  echo Run: python -m pip install -r requirements.txt
  pause
  exit /b 1
)

%PYTHON_EXE% -m uvicorn api.main:app --host %API_HOST% --port %API_PORT%

echo.
echo Service stopped. Press any key to close.
pause >nul
