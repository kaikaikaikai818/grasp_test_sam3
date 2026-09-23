# 轻量文字分割与 RealSense 三维定位

程序使用 CLIPSeg 在 CPU 上定位文字描述的目标，再使用 MobileSAM 在 NVIDIA GPU 上生成精细分割结果。当前主入口固定先跑通螺丝刀：D455（`215122257404`）固定粗定位，D435i（`215222074676`）手内精定位，全程只做视觉输出。

## 运行

打开 `run.py`，在文件顶部修改运行模式和目标文本，然后点击 VS Code 右上角运行按钮。

单张图片模式：

```python
RUN_MODE = "image"
IMAGE_PATH = r"test3.jpg"
TEXT_PROMPT = "a screwdriver"
```

双相机模式：

```python
RUN_MODE = "realsense"
TEXT_PROMPT = "a screwdriver"
```

双相机模式自动把 D455 作为固定全局相机、D435i 作为机械臂末端相机。两个实时窗口显示 mask、目标编号、深度和 XYZ；按 `Q` 或 `Esc` 退出。D435i 的 IMU 当前不使用。

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
