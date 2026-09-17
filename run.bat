@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONWARNINGS=ignore
set PYTHONDONTWRITEBYTECODE=1
set OPENCV_LOG_LEVEL=SILENT
"%~dp0.venv\Scripts\pythonw.exe" "%~dp0run.py"
