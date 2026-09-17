# 本地文字工具抓取入口

新增的 `grasp_tool.py` 不修改原来的 `grasp_cylinder.py`。它使用上一级
MobileSAM 项目的 CLIPSeg + MobileSAM，根据代码顶部的文字寻找工具。

```python
TEXT_PROMPT = "a wrench"
ENABLE_ROBOT_GRASP = False
```

第一次接真机时保持 `ENABLE_ROBOT_GRASP = False`，只检查两个窗口中的 mask、
深度和坐标。此时程序不会建立 UR5 控制连接，也不会初始化夹爪；D455 显示
基座坐标，D435i 显示相机坐标。确认两台相机定位都正确后，再配置具体工具的
夹持高度、方向和夹爪参数，最后才允许机械臂抓取。

- 原红色圆柱入口：`grasp_cylinder.py`
- 文字工具入口：`grasp_tool.py`

## 双相机稳定定位

`grasp_tool.py` 中的 `D455_ROI = (130, 80, 530, 420)` 是 D455 工作台裁剪范围，
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

当 D455 和 D435i 同时通过时，程序每 2 秒追加一条记录到：

```text
outputs/validation/measurements.jsonl
```

每行记录时间、提示词、置信度、像素中心、深度、角度、稳定波动、D455
基座坐标和 D435i 相机坐标。当前安全模式无法取得机械臂实时 TCP，因此
`cross_camera_same_target_verified` 和 `robot_motion_authorized` 都为 `false`。
视觉门控通过只表示坐标适合拿来测量比较，不代表允许机械臂运动。
