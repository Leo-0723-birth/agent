@echo off
chcp 65001 > nul
cd /d "%~dp0..\.."
.venv\Scripts\python.exe "官方要求交付物\04_API与运行脚本\api_service.py" 8787
pause
