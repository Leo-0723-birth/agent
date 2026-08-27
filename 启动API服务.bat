@echo off
title 监管问询扫雷预警系统 - FastAPI 服务
cd /d "%~dp0"

echo ============================================================
echo   监管问询扫雷预警系统（FastAPI + 高保真前端）
echo   启动完成后，下方将显示访问网址：
echo     Local     本机:   http://localhost:8000
echo     API 文档:         http://localhost:8000/docs
echo   关闭本窗口即停止服务。
echo ============================================================
echo.

".venv\Scripts\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000

echo.
echo 服务已停止。按任意键关闭窗口。
pause >nul
