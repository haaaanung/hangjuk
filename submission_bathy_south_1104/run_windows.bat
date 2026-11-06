@echo off
cd /d "%~dp0"
set /p "DATAZIP=공식 문제 데이터 ZIP 전체 경로: "
python predict.py --zip "%DATAZIP%"
pause
