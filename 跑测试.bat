@echo off
chcp 65001 >nul
cd /d D:\competition_agent

set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

echo 正在运行 pytest ...
.venv\Scripts\python.exe -m pytest backend\tests -q --tb=short > tests_report.txt 2>&1
echo 测试完成，结果已保存到 tests_report.txt
.venv\Scripts\python.exe -m pytest backend\tests -q --tb=short

pause
