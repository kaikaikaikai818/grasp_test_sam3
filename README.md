# 轻量文字分割与 RealSense 三维定位

程序使用 CLIPSeg 在 CPU 上定位文字描述的目标，再使用 MobileSAM 在 NVIDIA GPU 上生成精细分割结果。支持单张图片，也支持 D455 与 D435i 双 RealSense 实时识别和相机坐标系三维定位。

## 运行

打开 `run.py`，在文件顶部修改运行模式和目标文本，然后点击 VS Code 右上角运行按钮。

单张图片模式：

```python
RUN_MODE = "image"
IMAGE_PATH = r"test3.jpg"
TEXT_PROMPT = "a wrench"
```

双相机模式：

```python
RUN_MODE = "realsense"
TEXT_PROMPT = "a wrench"
```

双相机模式自动把 D455 作为固定全局相机、D435i 作为机械臂末端相机。两个实时窗口显示 mask、目标编号、深度和 XYZ；按 `Q` 或 `Esc` 退出。D435i 的 IMU 当前不使用。

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

## RealSense 输出

```text
outputs\realsense\d455\sam_result.jpg
outputs\realsense\d455\details\result.json
outputs\realsense\d435i\sam_result.jpg
outputs\realsense\d435i\details\result.json
```

每个目标还会在对应 `details` 文件夹保存独立 mask。JSON 包含相机型号、序列号、时间、目标框、分数、有效深度点数和 `[x, y, z]`。单位是米，X 向右、Y 向下、Z 向前。坐标属于各自相机，完成外参或手眼标定前不能直接作为机械臂基座坐标。
