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
