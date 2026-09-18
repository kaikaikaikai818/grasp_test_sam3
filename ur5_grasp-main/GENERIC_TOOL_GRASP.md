# 文字指定的通用工具抓取

在 `grasp_tool.py` 中只修改：

```python
TEXT_PROMPT = "a screwdriver"
```

程序不会按工具名称写死抓取点。它会要求 D455 与 D435i 各自只看到一个合格实例，再从 D435i 的 mask 和有效深度中找工具中段的平行夹爪候选。候选会避开两端；圆形、太短、太细、深度不足、夹爪宽度不合适或同时检测到多个同类目标时，画面会显示 `GRASP REJECTED`，不会发送运动命令。

## 首次实测配置

首次实体测试前，在 `grasp_tool.py` 填写以下值。未填写时识别和候选预览仍可用，但 `R` 键被锁定。

- `SCAN_TCP_POSE`：由示教器记录的六维 TCP 姿态。该姿态必须高于安全通行高度，且夹爪不会遮挡 D455。
- `GRIPPER_YAW_OFFSET_RAD`：TCP 朝下时，夹爪手指方向的实测偏角。
- `GRIPPER_LOWEST_POINT_OFFSET_M`：TCP 到夹爪最低点的竖直距离，填正数。
- `GRIPPER_CONTACT_OFFSET_M`：TCP 到手指接触面高度的竖直距离，填正数。
- `SUPPORT_TOP_Z_M`、`MIN_SUPPORT_CLEARANCE_M`：支撑件顶面高度和最小净空。
- `GRIPPER_MIN_JAW_WIDTH_M`、`GRIPPER_MAX_JAW_WIDTH_M`：可安全夹持的宽度范围。

第一版要求工具放在统一的垫高支撑件上，抓取中段悬空，并且工作区内只留一个与 prompt 相符的工具。

## 操作顺序

1. 默认 `ENABLE_SAFE_APPROACH_TEST = False`，此时只做识别和抓取候选预览。
2. 完成扫描姿态示教后，将 `ENABLE_SAFE_APPROACH_TEST = True`，重新运行，按小写 `s` 到扫描姿态。
3. 两台相机均为 `STABLE`、双相机关联通过、D435i 显示 `GRASP CANDIDATE` 后，按小写 `p` 到安全观察点。D455 随后被冻结，避免把移动中的夹爪误识别成工具。
4. 将 `ENABLE_SAFE_DESCENT_TEST = True` 后运行，按小写 `d`。它只会低速下降，并停在工具上方 40 mm；不会闭爪。
5. 仅在所有实测参数填写完、无接触测试成功，并将 `ENABLE_GENERIC_GRASP = True` 后，按小写 `r`。它会低速夹持、低力闭爪，且只测试抬升 20 mm。

任一步报错或候选发生变化，程序停止后续动作；不要通过放宽工作空间、跳过扫描姿态或填写估计值来绕过锁定。
