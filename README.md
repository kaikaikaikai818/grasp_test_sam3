# 双相机工具抓取系统

程序使用 D455 固定相机粗定位、D435i 腕部相机精定位，并通过共享的标定、安全门和工具策略控制 UR5 与夹爪。

## 主程序

双击根目录的 `run.bat`，选择工具即可。当前状态：

- 螺丝刀：已经用大、小两把实机验证，主入口使用自动流程；D455 稳定并确认路径清空后按一次小写 `a`，后续自动执行。
- 卷尺：已经能够抓取，主入口暂时保留 `P → Y → D → R` 最终验收流程；连续两次回归通过后切换为自动流程。
- 活动扳手、钳子、胶带切割器：尚未完成安全验收，主入口不会连接机械臂运动。

`screwdriver_workflow.bat` 和 `tape_measure_workflow.bat` 是维护、标定和故障诊断入口，不是日常主程序。

## 视觉预览

不需要机器人运动时，可运行 `scripts/visual_preview.py`。这个维护工具只显示分割、深度和三维坐标，不连接机械臂或夹爪；运行模式和目标文字在文件顶部设置。

视觉预览自动把 D455 作为固定全局相机、D435i 作为机械臂末端相机。两个实时窗口显示 mask、目标编号、深度和 XYZ；按 `Q` 或 `Esc` 退出。D435i 的 IMU 当前不使用。

D455会先裁剪 `D455_ROI = (170, 95, 500, 370)` 指定的工作台区域，再放大识别远处的小螺丝刀；蓝框就是当前搜索范围。识别结果的mask、像素坐标和XYZ会还原到D455完整画面。黄色矩形是只用于预览的螺丝刀手柄抓取框，红点是抓取中心；它不会产生机器人运动。

处理成功后会自动打开最终结果图片，不在终端输出模型日志。

## 输出

```text
outputs\图片名称\
├── sam_result.jpg
└── details\
    ├── sam_mask_001.png
    ├── sam_mask_002.png
    └── sam_result.json
```

- `sam_result.jpg`：最终预览图。
- `details/sam_mask_*.png`：每个目标的独立二值 mask。
- `details/sam_result.json`：目标框、置信度、耗时和显存数据，供后续评测与蒸馏使用。

重复处理同名图片时，该图片原有的详情结果会先清理，不会混入旧 mask。

当前环境要求 NVIDIA CUDA，MobileSAM 权重位于 `weights/mobile_sam.pt`。

## 螺丝刀双相机一致性验证

需要进入机器人项目验证已有标定时，双击根目录的 `screwdriver_workflow.bat`。
菜单会依次引导完成两轮五位置静止采集、候选补偿和独立验收。生成的
`ur5_grasp-main/camera_alignment.json` 绑定两台相机、分辨率及三份核心标定
文件，所有工具共用一次验证结果；换工具不需要重做。菜单 1 到 4 不发送机器人
运动或夹爪命令，菜单 5 才开放按键触发的 250 mm 以上高位观察点测试。

## RealSense 输出

```text
outputs\realsense\d455\sam_result.jpg
outputs\realsense\d455\details\result.json
outputs\realsense\d435i\sam_result.jpg
outputs\realsense\d435i\details\result.json
```

每个目标还会在对应 `details` 文件夹保存独立 mask。JSON 包含相机型号、序列号、时间、目标框、分数、有效深度点数和 `[x, y, z]`。单位是米，X 向右、Y 向下、Z 向前。坐标属于各自相机，完成外参或手眼标定前不能直接作为机械臂基座坐标。
