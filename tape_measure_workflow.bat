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
echo 4. Calibrate tape-measure gripper opening - gripper only
echo 5. Test tape-measure oriented no-contact descent - press P, Y, then D
echo 6. Test one oriented low-force tape-measure grasp - press P, Y, D, then R
echo Q. Exit
set "ACTION="
set /p "ACTION=Select an option: "

if /i "%ACTION%"=="Q" exit /b 0
if "%ACTION%"=="1" goto vision
if "%ACTION%"=="2" goto observe
if "%ACTION%"=="3" goto recovery_lift
if "%ACTION%"=="4" goto gripper_opening
if "%ACTION%"=="5" goto no_contact_descent
if "%ACTION%"=="6" goto low_force_grasp
echo [ERROR] Invalid option. Enter 1, 2, 3, 4, 5, 6, or Q.
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

:gripper_opening
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [GRIPPER ONLY] Robot motion control and cameras stay disconnected.
echo Keep the gripper clear. Smaller position values open it wider.
echo The verified tape-measure opening position is 4000.
echo Enter 4000 to open, then enter Q to exit.
echo Stop when the opening is slightly wider than the tape measure; enter Q to exit.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --gripper-test
pause
goto menu

:no_contact_descent
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode connects robot control but does not initialize the gripper.
echo Keep the path clear and confirm the active TCP is TCP_clamp.
echo Wait for both cameras to show PASS, press P, then wait for BASE AXIS STABLE.
echo Press Y at 250 mm to align the gripper, wait for D435i to stabilize again,
echo then press D only when the preview says SAFE DESCENT READY.
echo The robot stops 40 mm above the calculated tape-body midpoint and will not close the gripper.
echo Press U to recover vertically to 250 mm, then press Q to exit.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "tape measure" --stage descent --check-calib
pause
goto menu

:low_force_grasp
call :require_robot_modules
if errorlevel 1 goto missing_modules
echo [WARNING] This mode controls both the robot and gripper for one guarded grasp.
echo Confirm TCP_clamp is active, the tape measure is centered, and the path is clear.
echo Wait for both cameras to show PASS, press P, then wait for BASE AXIS STABLE.
echo Press Y at 250 mm, wait for D435i to stabilize again, then press D.
echo Visually confirm the 40 mm no-contact stop before pressing R.
echo R opens to 4000, descends to the measured body midpoint, closes at force 20,
echo checks contact torque, and lifts only 50 mm. A failed contact check opens and retreats.
echo After success press O to release, U to recover to 250 mm, then Q to exit.
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "tape measure" --stage grasp --tape-grasp-test --check-calib
pause
goto menu

:missing_modules
echo [ERROR] Required modules are missing. Run screwdriver_workflow.bat option 0 once.
pause
goto menu
