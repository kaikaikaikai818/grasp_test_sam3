# 平放螺丝刀：标定与自适应抓取

流程用 D435i 深度拟合支撑面（桌面或底板），并用“投影宽度估出的手柄直径 + 一次性夹爪偏移”自动算出夹持高度，因此换不同大小的螺丝刀不需要重标定。

## 一次性标定

在 VS Code 的 `(sam3)` 终端执行：

```powershell
python ur5_grasp-main\calibrate_screwdriver_grasp.py plane
```

移开螺丝刀，让相机只看空支撑面（桌面/底板）。点击画面后按小写 `s` 保存。这个程序只读相机和 TCP，不会移动机械臂或控制夹爪。

放回螺丝刀，用示教器把**张开的夹爪**手动放到螺丝刀手柄中段的目标夹持高度。然后执行：

```powershell
python ur5_grasp-main\calibrate_screwdriver_grasp.py contact
```

当画面显示 `HANDLE STABLE`，确认绿色 mask 盖住手柄（状态行 `top/median` 相差约一个手柄半径、`D~` 接近真实直径），点击画面后按小写 `k` 保存。脚本会估算手柄直径并计算“夹爪偏移”；若估算直径不在 10–100mm 会拒绝保存。配置保存在本机 `grasp_surface_calibration.json`，不会上传 GitHub。

## 抓取测试

在 `grasp_tool.py` 中设置：

```python
TEXT_PROMPT = "a screwdriver"
ENABLE_ROBOT_GRASP = False
ENABLE_SAFE_APPROACH_TEST = True
ENABLE_SAFE_DESCENT_TEST = True
ENABLE_SCREWDRIVER_GRASP = True
ENABLE_ONE_KEY_GRASP = True
```

重新运行程序。

**一键抓取（推荐）**

1. 机械臂停在安全位、工具平放在支撑面；等 D455 显示 `STABLE`、`PASS`。
2. 按小写 `a`：用 D455 对齐坐标粗定位到目标上方，随后自动完成无接触下降与夹持。
3. 等 D435i 稳定后程序自动下降、低力闭爪（力 30）、读力矩确认，抬升 50 mm 后停住。
4. 按小写 `o` 松开，`c` 再闭合。

任一步超时（8 秒）或未通过深度、平面、位置、净空、姿态检查时，程序会停在安全位置，不会盲降。

**手动分步（调试用）**

`p` 到双相机安全观察点 -> `d` 无接触下降（停在夹持点上方 40 mm）-> `r` 夹持并抬升。
