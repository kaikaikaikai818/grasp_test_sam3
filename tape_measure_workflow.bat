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
echo ===== Tape measure dual-camera workflow =====
echo 1. Visual detection only - robot stays still
echo 2. Test high observation pose - press P when both cameras pass
echo 3. Recovery lift - press U to move vertically to 250 mm
echo Q. Exit
set "ACTION="
set /p "ACTION=Select an option: "

if /i "%ACTION%"=="Q" exit /b 0
if "%ACTION%"=="1" goto vision
if "%ACTION%"=="2" goto observe
if "%ACTION%"=="3" goto recovery_lift
echo [ERROR] Invalid option. Enter 1, 2, 3, or Q.
goto menu

:require_robot_modules
"%PROJECT_PYTHON%" -c "import rtde_receive, minimalmodbus" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:vision
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [SAFE MODE] Both cameras run; robot and gripper stay disconnected.
echo Put one tape measure on the fixed support surface and inspect both masks.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "tape measure" --stage vision --check-calib
pause
goto menu

:observe
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode connects robot control.
echo Press P only when both cameras show PASS and the path is clear.
echo The robot moves only to the 250 mm observation pose.
echo This mode does not descend and does not initialize the gripper.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "tape measure" --stage observe --check-calib
pause
goto menu

:recovery_lift
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] Press U once to keep XY and orientation and lift to 250 mm.
echo No horizontal motion, descent, or gripper command is allowed.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "tape measure" --stage observe --check-calib
pause
goto menu

:missing_modules
echo [ERROR] Required modules are missing. Run screwdriver_workflow.bat option 0 once.
pause
goto menu
