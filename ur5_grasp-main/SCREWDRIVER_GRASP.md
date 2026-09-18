# 平放螺丝刀：深度标定与低力抓取

本流程使用 D435i 深度自动计算纸箱顶面和螺丝刀手柄表面高度。首次标定完成后，不必每次用尺测量高度。

## 一次性标定

在 VS Code 的 `(sam3)` 终端执行：

```powershell
python ur5_grasp-main\calibrate_screwdriver_grasp.py plane
```

移开螺丝刀，让相机只看空纸箱顶面。点击画面后按小写 `s` 保存。这个程序只读相机和 TCP，不会移动机械臂或控制夹爪。

放回螺丝刀，用示教器把**张开的夹爪**手动放到螺丝刀手柄中段的目标夹持高度。然后执行：

```powershell
python ur5_grasp-main\calibrate_screwdriver_grasp.py contact
```

当画面显示 `HANDLE STABLE`，点击画面后按小写 `k` 保存 TCP 接触偏移。配置保存在本机 `grasp_surface_calibration.json`，不会上传 GitHub。

## 抓取测试

在 `grasp_tool.py` 中设置：

```python
TEXT_PROMPT = "a screwdriver"
ENABLE_ROBOT_GRASP = False
ENABLE_SAFE_APPROACH_TEST = True
ENABLE_SAFE_DESCENT_TEST = True
ENABLE_SCREWDRIVER_GRASP = True
```

重新运行程序，依次执行：

1. 等两台相机显示 `STABLE`、`PASS`；程序会用手柄最厚区域而不是螺丝刀整体中心定位。
2. 点击相机画面，按小写 `p`，移动到安全观察点。
3. 等 D435i 再次稳定，按小写 `d`，低速停在手柄上方 40 mm。
4. 目标和纸箱平面仍稳定时，按小写 `r`。程序低速下降到标定高度、低力闭爪，并且只抬升 20 mm。

任一深度、平面、位置或净空检查不通过时，程序拒绝执行 `r`。低力夹持未确认时，程序会张开夹爪并退回 40 mm 高度。
