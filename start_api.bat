@echo off
chcp 65001 >nul
title FastAPI Service - Risk Early Warning System
cd /d "%~dp0"

set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
:: 比赛演示：启用 mock 填充层补齐后端未完全产出的展示字段
set ENABLE_MOCK_FILL=1

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

".venv\Scripts\python.exe" -m uvicorn api.main:app --host %API_HOST% --port %API_PORT%

echo.
echo Service stopped. Press any key to close.
pause >nul
