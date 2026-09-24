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
echo 5. Test the high observation pose - press P when both cameras pass
echo 6. Test high safe orientation - no descent or gripper action
echo 7. Test no-contact descent - stop 40 mm above the tool
echo 8. Recovery lift - press U to move vertically to 250 mm
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
if "%ACTION%"=="6" goto rotate
if "%ACTION%"=="7" goto descent
if "%ACTION%"=="8" goto recovery_lift
echo [ERROR] Invalid option. Enter 0, 1, 2, 3, 4, 5, 6, 7, 8, or Q.
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
echo Press P when both cameras show PASS and the path is clear.
echo Press A only when D435i cannot initially see the tool.
echo This mode does not descend and does not control the gripper.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage observe --check-calib
pause
goto menu

:rotate
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode connects robot control.
echo First press P at dual-camera PASS, then wait for D435i STABLE.
echo Press Y once to align orientation at or above 250 mm.
echo This mode does not descend and does not control the gripper.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage rotate --check-calib
pause
goto menu

:descent
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode connects robot control and performs a slow descent.
echo Press P at dual-camera PASS, then Y after D435i becomes STABLE.
echo Wait for SAFE DESCENT READY 3/3, then press D once.
echo The TCP stops 40 mm above the tool. The gripper is not initialized.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage descent --check-calib
pause
goto menu

:recovery_lift
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode connects robot control.
echo Press U once to keep the current XY and orientation and lift to 250 mm.
echo No horizontal motion, descent, or gripper command is allowed.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage observe --check-calib
pause
goto menu

:missing_modules
echo [ERROR] Required modules are missing. Select option 0 first.
pause
goto menu
