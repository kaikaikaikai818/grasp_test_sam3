@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements-deploy.txt
if errorlevel 1 goto :error
echo 环境安装完成。
pause
exit /b 0

:error
echo 环境安装失败，请检查 Python 3.12 和网络连接。
pause
exit /b 1
