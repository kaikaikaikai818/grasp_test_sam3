@echo off
setlocal
cd /d "%~dp0"
python -c "import sys; assert sys.version_info[:2] == (3,12), 'Please install 64-bit Python 3.12'"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe check_environment.py
if errorlevel 1 exit /b 1
echo Environment ready. Edit config.yaml, then run demo_image.bat.
pause
