@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" set "PROJECT_PYTHON=%~dp0SAM_GRConvNet\.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
  echo [ERROR] Project virtual environment was not found.
  echo Run setup_env.bat or SAM_GRConvNet\setup_3060.bat first.
  pause
  exit /b 1
)

:menu
echo.
echo ===== Screwdriver dual-camera workflow =====
echo 0. Install read-only robot dependencies - once only
echo 1. Collect first five-position dataset - robot stays still
echo 2. Build one shared alignment candidate - no robot motion
echo 3. Collect second independent dataset - robot stays still
echo 4. Validate with the independent dataset
echo 5. Test the high observation pose - press A to move
echo Q. Exit
set "ACTION="
set /p "ACTION=Select an option: "

if /i "%ACTION%"=="Q" exit /b 0
if "%ACTION%"=="0" goto setup
if "%ACTION%"=="1" goto collect_fit
if "%ACTION%"=="2" goto fit
if "%ACTION%"=="3" goto collect_validate
if "%ACTION%"=="4" goto validate
if "%ACTION%"=="5" goto observe
echo [ERROR] Invalid option. Enter 0, 1, 2, 3, 4, 5, or Q.
goto menu

:require_robot_modules
"%PROJECT_PYTHON%" -c "import rtde_receive, minimalmodbus" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:setup
"%PROJECT_PYTHON%" -m pip install -r "%~dp0requirements-robot-validation.txt"
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
) else (
  echo [OK] Read-only robot dependencies are installed in the project environment.
)
pause
goto menu

:collect_fit
call :require_robot_modules
if errorlevel 1 goto missing_modules
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage vision --check-calib
pause
goto menu

:fit
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\calibrate_camera_alignment.py" fit
pause
goto menu

:collect_validate
call :require_robot_modules
if errorlevel 1 goto missing_modules
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage vision --check-calib
pause
goto menu

:validate
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\calibrate_camera_alignment.py" validate
pause
goto menu

:observe
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode connects robot control.
echo Press A only after the camera view is stable and the path is clear.
echo This mode does not descend and does not control the gripper.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage observe --check-calib
pause
goto menu

:missing_modules
echo [ERROR] Required modules are missing. Select option 0 first.
pause
goto menu
