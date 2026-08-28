@echo off
chcp 65001 >nul
cd /d "%~dp0"

set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

if "%API_PORT%"=="" set API_PORT=8000
if "%API_HOST%"=="" set API_HOST=0.0.0.0

echo Starting API service on port %API_PORT% ...

:: Run uvicorn in a separate window so this script can reliably wait and open browser
start "API Server" cmd /c "".venv\Scripts\python.exe" -m uvicorn api.main:app --host %API_HOST% --port %API_PORT%"

echo Waiting for service to be ready...
:wait_loop
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%API_PORT%/api/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch { } exit 1"
if %errorlevel%==0 goto service_ready
ping -n 2 127.0.0.1 >nul
goto wait_loop

:service_ready
echo Service is ready. Opening browser...
start "" "http://127.0.0.1:%API_PORT%"
echo.
echo API is running at: http://127.0.0.1:%API_PORT%
echo API Docs:          http://127.0.0.1:%API_PORT%/docs
echo Close the API Server window to stop the service.
echo.
pause
