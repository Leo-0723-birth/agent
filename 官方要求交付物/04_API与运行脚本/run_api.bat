@echo off
chcp 65001 > nul
cd /d "%~dp0..\.."
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
%PYTHON_EXE% "官方要求交付物\04_API与运行脚本\api_service.py" 8787
pause
