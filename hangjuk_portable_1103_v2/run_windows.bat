@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo 공식 문제 데이터_수정_1103_v2.zip 파일을 이 bat 파일 위로 끌어 놓으세요.
  pause
  exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.12를 설치하고 PATH 옵션을 켜주세요.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo 패키지 설치 실패. 위 오류를 확인하세요.
  pause
  exit /b 1
)
python -X utf8 -u predict.py --zip "%~1"
if errorlevel 1 (
  echo 실행 실패. 위 오류를 확인하세요.
  pause
  exit /b 1
)
echo 완료: output\submission.xlsx
pause
