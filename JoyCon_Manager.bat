@echo off
rem Joy-Con link manager launcher (double-click). Works from any folder: paths are relative to this file.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto no_env
rem A .venv copied from another PC points at that PC's Python and fails silently: check it really works.
".venv\Scripts\python.exe" -c "import teachabot.btlink.manager" >nul 2>nul
if errorlevel 1 goto broken_env
goto run

:broken_env
echo [조이콘 연결 관리자] 이 폴더의 .venv 가 이 PC에서 동작하지 않습니다.
echo 다른 PC에서 복사해 왔거나, 필요한 패키지가 빠져 있습니다.
echo.
where uv >nul 2>nul
if errorlevel 1 goto no_uv
choice /C YN /M ".venv 를 지우고 이 PC에 맞게 다시 설치할까요? 몇 분 걸립니다"
if errorlevel 2 goto manual
rmdir /s /q ".venv"
goto install

:no_env
echo [조이콘 연결 관리자] 이 PC에는 아직 파이썬 환경 .venv 가 없습니다.
echo.
where uv >nul 2>nul
if errorlevel 1 goto no_uv
choice /C YN /M "지금 설치할까요? 처음 한 번만 필요하며 몇 분 걸립니다"
if errorlevel 2 goto manual

:install
uv venv --python 3.12 .venv
if errorlevel 1 goto failed
uv pip install --python .venv\Scripts\python.exe -e .
if errorlevel 1 goto failed
echo.
echo 설치 완료. 관리자를 실행합니다.
goto run

:no_uv
echo uv 가 설치돼 있지 않습니다. 먼저 uv 를 설치해 주세요: https://docs.astral.sh/uv/
echo.
:manual
echo 설치 방법: 이 폴더에서 아래 줄을 실행한 뒤 다시 더블클릭하세요. 이미 .venv 가 있으면 먼저 지웁니다.
echo    uv venv --python 3.12 .venv
echo    uv pip install --python .venv\Scripts\python.exe -e .
echo.
pause
exit /b 1

:failed
echo.
echo 설치 중 오류가 났습니다. 위 메시지를 확인해 주세요.
pause
exit /b 1

:run
start "" ".venv\Scripts\pythonw.exe" "scripts\joycon_manager.py" %*
exit /b 0
