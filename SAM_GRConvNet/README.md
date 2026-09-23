# 文字引导的平面抓取项目

本项目把文字目标分割与二维抓取预测组合为一条独立视觉链路：

```text
图片或 RealSense → 文字目标解析 → CLIPSeg 目标候选 → MobileSAM 完整物体
                  → CLIP 余弦类别确认 → mask 几何抓取 → 抓取候选与可视化
```

CLIPSeg 只提供目标上的语义种子点。默认的 `complete_object: true` 会让 MobileSAM
在多个尺度的候选中选择完整且不属于整片背景的物体轮廓，避免长柄工具只保留最显著的头部。

当前版本只输出研究用视觉结果，永远不会连接或控制 UR5。几何后端根据完整 mask 的主轴、局部厚度和安全边界计算平面抓取，适合先验证桌面上的扳手、螺丝刀和钳子。接入机器人前仍必须通过批量旋转一致性和 RealSense 深度验收。

## 多工具选择与拒绝

`config.yaml` 的 `recognition.categories` 定义了可抓取工具、中文/英文别名和只用于排除的干扰类别。当前可指定活动扳手、螺丝刀、钳子、锤子和内六角；笔、手机和盒子参与比较，但不能被请求为抓取目标。

程序只把指定工具名称交给 CLIPSeg 产生候选，不再横向比较不同文字的分割热力图。MobileSAM取得完整mask后，程序增加10%留白裁剪物体，把mask外区域替换为中性灰色，再使用同一权重内部的CLIP图像与文本编码器计算归一化余弦相似度。每个类别使用三条文本模板并平均特征。

目标类别必须排名第一，同时达到 `min_similarity` 且领先第二名至少 `min_margin`，才会进入抓取几何计算。否则画面显示红色候选和 `REJECT` 原因。`result.json` 记录实际定位prompt、全部分类模板、`semantic_backend: clip_cosine`、胜出类别、全部类别相似度和拒绝原因。画面中的 `C` 是类别余弦相似度，`Q` 是抓取几何质量，两者含义不同。

例如，寻找螺丝刀可以使用中文或英文：

```powershell
.\.venv\Scripts\python.exe run_realsense.py `
  --text "螺丝刀" `
  --serial "215222074676" `
  --output "outputs\realsense_screwdriver"
```

按 `G` 推理一次。如果画面中只有扳手或笔，正确结果应为语义通过数量0。终端会分别显示候选数量、语义通过数量和安全抓取数量，并列出每个候选的全部余弦相似度；只有最后一项大于0才会出现可用抓取框。无训练阶段默认 `min_similarity: 0.20`、`min_margin: 0.01`，真机混合测试后只校准这两个配置值。此比较机制是流程验证阶段的拒绝门，尚未达到机器人运动授权标准。

## 目录和共享权重

项目自带 MobileSAM 代码。以下两个模型通过 `config.yaml` 共享，不复制：

- `../weights/mobile_sam.pt`，约 39 MB；
- `../weights/clipseg-rd64-refined/`，约 575 MB。

如果迁移到另一台电脑，编辑 `config.yaml` 中的两个 SAM 路径。相对路径以 `config.yaml` 所在目录为基准，也可以填写 Windows 绝对路径。

## RTX 3060 Ti 环境

要求 Windows 10/11 64 位、Python 3.12 64 位和可用的 NVIDIA 驱动。双击 `setup_3060.bat` 会创建本目录专用 `.venv`，安装 PyTorch 2.7.1/cu118、torchvision 0.22.1 及 `requirements.txt` 中的固定版本，然后验证 CUDA、显卡和两个分割模型路径。无需单独安装 CUDA Toolkit，也不会修改 LGD 环境。

安装完成后运行单图演示：

```powershell
.\.venv\Scripts\python.exe run_image.py `
  --image "..\test.jpg" `
  --text "grasp the wrench" `
  --output "outputs\wrench"
```

也可以双击 `demo_image.bat`。输出包括：

- `grasp_overlay.jpg`：SAM mask、接受或拒绝状态及抓取框；
- `quality.png`、`angle.png`、`width.png`：三张预测图；
- `quality_masked.png`：mask 约束后的质量图；
- `maps.npz`：原始浮点预测图；
- `result.json`：抓取中心、角度、宽度、质量、拒绝原因和模型信息。

## RealSense 安全预览

运行：

```powershell
.\.venv\Scripts\python.exe run_realsense.py --text "grasp the wrench"
```

窗口默认只显示实时画面；按 `G` 对当前帧运行一次完整推理并保存 RGB、对齐深度和结果，按 `Q` 或 `Esc` 退出。可用 `--serial` 指定 D435i 或 D455。该入口没有机器人网络或运动接口。

## 十张图片验收

准备至少 10 张同类工具图片后运行：

```powershell
.\.venv\Scripts\python.exe validate_folder.py `
  --images "D:\grasp_test_images" `
  --text "grasp the wrench"
```

程序对每张原图和顺时针旋转 90° 的版本重新执行 SAM 与抓取推理。只有全部图像均产生可接受候选，且旋转轴向误差中位数不超过 15°，`passes_rotation_gate` 才为 `true`。无论结果如何，报告中的 `robot_motion_authorized` 始终为 `false`。

异常结果会被拒绝，包括：mask 过小、质量低、角度向量过弱、宽度非有限、宽度超过目标 mask 短边的 1.25 倍，或超过候选点局部目标厚度的 1.5 倍。mask 外像素不能参与最高点选择。

## LGD 空间清单

在 3060 Ti 电脑上先执行只读盘点：

```powershell
python inventory_lgd.py "D:\zky\LGD" --output lgd_inventory.json
```

脚本只统计文件，不删除内容。重点人工核对 LGD diffusion 主权重、LGD checkpoint、Grasp-Anything 数据集、`exp/runs/logs/wandb` 和旧虚拟环境。保留原始图片、RealSense 数据、标定文件、手眼外参、实验记录及 MobileSAM 共享权重。确认新项目验证完成后，再按清单逐项删除；不要按 `.pt` 或 `.pth` 后缀批量删除。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖完整物体 mask 选择、几何抓取中心、mask 外高分排除、小 mask、弱角度、异常宽度以及轴向角度的 180° 周期。

旧 PromptGD 权重在实拍扳手上持续输出异常宽度，现已从正式流程移除。历史实现仍可从 Git 记录查阅，不再占用主程序结构。
