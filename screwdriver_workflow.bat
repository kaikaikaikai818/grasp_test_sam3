@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" set "PROJECT_PYTHON=%~dp0SAM_GRConvNet\.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
  echo [错误] 没有找到项目虚拟环境。请先运行 setup_env.bat 或 SAM_GRConvNet\setup_3060.bat。
  pause
  exit /b 1
)

:menu
echo.
echo ===== 螺丝刀双相机流程 =====
echo 0. 安装机器人只读验证依赖（只需一次）
echo 1. 采集第一轮五位置数据（机器人静止）
echo 2. 生成一次性候选补偿（仍禁止运动）
echo 3. 采集第二轮独立五位置数据（机器人静止）
echo 4. 独立验收候选补偿
echo 5. 高位观察点测试（会连接机器人；按 a 才运动，不下降、不控制夹爪）
echo Q. 退出
set /p "ACTION=请选择: "

if /i "%ACTION%"=="Q" exit /b 0
if "%ACTION%"=="0" goto setup
if "%ACTION%"=="1" goto collect_fit
if "%ACTION%"=="2" goto fit
if "%ACTION%"=="3" goto collect_validate
if "%ACTION%"=="4" goto validate
if "%ACTION%"=="5" goto observe
echo 输入无效。
goto menu

:require_robot_modules
"%PROJECT_PYTHON%" -c "import rtde_receive, minimalmodbus" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:setup
"%PROJECT_PYTHON%" -m pip install -r "%~dp0requirements-robot-validation.txt"
if errorlevel 1 (
  echo [失败] 依赖安装失败。
) else (
  echo [完成] 只读验证依赖已安装在项目虚拟环境中。
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
echo [注意] 该模式会连接机器人控制。只有相机窗口显示稳定且路径无障碍时才按小写 a。
"%PROJECT_PYTHON%" "%~dp0ur5_grasp-main\grasp_tool.py" --prompt "screwdriver" --stage observe --check-calib
pause
goto menu

:missing_modules
echo [缺少依赖] 请先选择菜单 0 安装机器人只读验证依赖。
pause
goto menu
