@echo off
title 上市公司监管问询扫雷预警系统 - 导航入口
cd /d "%~dp0"

echo ============================================================
echo   上市公司监管问询扫雷预警系统（导航入口）
echo   启动完成后，下方将显示访问网址：
echo     Local     本机:   http://localhost:8501
echo     Network   局域网: http://192.168.x.x:8501 （同一局域网用户可访问）
echo     External  外网:   http://公网IP:8501 （需端口映射/防火墙放行）
echo   关闭本窗口即停止服务。
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run 导航入口.py

echo.
echo 服务已停止。按任意键关闭窗口。
pause >nul