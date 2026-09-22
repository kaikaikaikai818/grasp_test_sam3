@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe run_image.py --image "..\test.jpg" --text "grasp the wrench" --output "outputs\wrench"
if errorlevel 1 pause & exit /b 1
start "" "outputs\wrench\grasp_overlay.jpg"
pause
