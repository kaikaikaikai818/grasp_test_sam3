@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONWARNINGS=ignore
set PYTHONDONTWRITEBYTECODE=1
set OPENCV_LOG_LEVEL=SILENT
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" set "PROJECT_PYTHON=%~dp0SAM_GRConvNet\.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
  echo [ERROR] Project virtual environment not found.
  echo Run setup_env.bat or SAM_GRConvNet\setup_3060.bat first.
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" "%~dp0run.py"
if errorlevel 1 pause
