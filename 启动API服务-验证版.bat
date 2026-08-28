@echo off
chcp 65001 >nul
cd /d D:\competition_agent

:: 限制 BLAS 线程，避免启动就吃满内存
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

:: 启动 API 服务（uvicorn），并自动打开浏览器看前端页面
echo 启动 API 服务，稍后会自动打开 http://127.0.0.1:8080 ...
start "" "http://127.0.0.1:8080"

.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8080

pause
