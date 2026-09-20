# 本地文字工具抓取入口

`grasp_tool.py` 使用上一级 MobileSAM 项目的 CLIPSeg + MobileSAM，根据代码顶部的
文字寻找工具，并完成 D455 全局粗定位 -> D435i 腕部精定位 -> 自适应夹持。

```python
TEXT_PROMPT = "a screwdriver"
ENABLE_ROBOT_GRASP = False
```

第一次接真机时保持 `ENABLE_ROBOT_GRASP = False`，只检查两个窗口中的 mask、
深度和坐标。确认两台相机定位都正确后，再配置具体工具的夹持参数，最后才允许
机械臂抓取。

## 一键抓取

`ENABLE_ONE_KEY_GRASP = True` 时，按 `a` 用 D455 对齐坐标粗定位到目标上方，
随后自动完成无接触下降与夹持：等 D435i 连续稳定后下降到“夹持点上方 40mm”，
再下降到夹持高度、低力闭爪、读力矩确认、抬升 `SCREWDRIVER_TEST_LIFT_M` 后停住。
任一步超时（`AUTO_GRASP_TIMEOUT_S`）或未通过安全门都会停在安全位置，不会盲降。
完成后按 `o` 松开、`c` 再闭合。手动仍可用 `P` / `D` / `R` 分步执行。

## 双相机稳定定位

`grasp_tool.py` 中的 `D455_ROI = (170, 95, 500, 370)` 是 D455 工作台裁剪范围，
格式为 `(左, 上, 右, 下)`。D455 会放大该区域识别远处小工具。画面状态含义：

- `SEARCHING`：没有连续有效目标；
- `TRACKING`：已经发现目标，但中心或深度仍不稳定；
- `STABLE`：连续多帧稳定，才会发布坐标。

纯视觉测试继续保持 `ENABLE_ROBOT_GRASP = False`。退出时程序会主动释放两台相机。

## 坐标验收与安全门控

稳定跟踪之后程序还会检查置信度、检测框尺寸、有效深度点数量，以及 D455
输出是否位于 `WORKSPACE_LIMITS`。窗口中的含义如下：

- `PASS`：该相机结果通过视觉验收门槛；
- `REJECT`：结果虽然可能稳定，但没有通过验收，第三行会显示原因；
- `validation gate: PASS`：坐标会进入测量日志。

启动后先移动工具到测试位置，再按数字键标记：

- `1`：圆盘中心；
- `2`：画面左侧；
- `3`：画面右侧；
- `4`：画面上方；
- `5`：画面下方。

按键后程序会清除上一位置的稳定历史，重新等待两台相机达到 `STABLE + PASS`。
窗口标题中的 `CENTER:3` 表示中心位置已经记录3条。没有按数字键时不会写入
测量数据。

当 D455 和 D435i 同时通过时，程序每 2 秒追加一条记录到本次运行的独立文件：

```text
outputs/validation/measurements_年月日_时分秒.jsonl
```

每行记录会带有 `session_id` 和 `position_label`，同时保存时间、提示词、
置信度、像素中心、深度、角度、稳定波动、D455
基座坐标和 D435i 相机坐标。当前安全模式无法取得机械臂实时 TCP，因此
`cross_camera_same_target_verified` 和 `robot_motion_authorized` 都为 `false`。
视觉门控通过只表示坐标适合拿来测量比较，不代表允许机械臂运动。

## 双相机目标关联预演

`ENABLE_ROBOT_STATE_READ = True` 通过独立的 `read_only_robot_state.py` 建立
RTDE 状态接收连接，只读取当前 TCP 位姿。厂家提供的 `UR_Robot.py` 保持原样；
程序不会创建 `RTDEControlInterface`，`ENABLE_ROBOT_GRASP = False` 仍然锁定
所有机械臂运动和夹爪操作。

D435i 检出的相机坐标会通过 `cam2end_20260906.txt` 和实时 TCP 转换成基座坐标，
再与 D455 的基座坐标比较：

- `SAME TARGET`：两者距离不超过 `TARGET_ASSOCIATION_MAX_DISTANCE_M`；
- `TARGET MISMATCH`：两者距离超过阈值；
- `ASSOCIATION WAITING`：检测门控未通过，或者只读 TCP 不可用。

关联通过时只计算并记录目标坐标和目标上方 `APPROACH_HEIGHT_M` 的观察点，窗口
会标记 `preview only`。日志中的 `robot_motion_authorized` 始终为 `false`。
如果机械臂网络不可达，程序仍会运行两个相机窗口，但 D435i 只显示相机坐标，
不会进行跨相机关联。

## 双相机相对校正

先在中心、左、右、上、下五个位置各记录至少3条有效数据，然后打开并运行
`calibrate_camera_alignment.py`。脚本会自动选择 `outputs/validation` 中最新的
`measurements_*.jsonl`，对每个位置取中位数，并生成：

```text
ur5_grasp-main/camera_alignment.json
```

主程序下次启动时会自动加载该文件。D455窗口显示 `aligned` 坐标，关联状态
显示 `aligned delta`；日志同时保存原始 `base_xyz_m` 和校正后的
`base_xyz_aligned_m`。校正只用于双相机目标关联，不覆盖原始坐标，也不授权
机械臂运动。应使用一轮新的位置数据验证校正效果，不能只看拟合数据。
