#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文字指定工具抓取原型（手外粗定位 -> 手内精定位 -> 识别抓取，含夹爪）

全流程（按 g 触发）：
  ① 粗定位  手外 D455 检测红色圆柱 -> pixel_to_robot_coords -> 基座坐标
            -> TCP 垂直升到安全高度 -> 自动摆正并验证姿态 -> 粗移到目标上方
  ② 精定位  手内 D435I 检测 -> pixel_to_base -> 细化基座坐标（多帧重试）
            -> 精对齐到细化坐标上方；若多次失败则立即中止，禁止下降抓取
  ③ 抓取   下降 -> 收爪 -> 抬起（只抓起+抬起，不含放置）

按键：
  g 执行① ② ③     t 仅升高并摆正     o 夹爪张开     c 夹爪闭合     q 退出
自检：
  --check-calib  启动时读两台相机实时内参，与标定内参比对，超阈值拒绝执行
  --gripper-test 交互式校定夹爪开/合 position
"""
import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
warnings.filterwarnings("ignore")
import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot, load_camera_ini
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.camera_bsp.hand_out_eye_calibration import HandOutEyeCalibration
from bsp.camera_bsp.sam_tool_detect import SamToolDetector, TemporalResultFilter

# -------------------------- 配置 --------------------------
SCRIPT_ROOT = Path(__file__).resolve().parent
TEXT_PROMPT = "a wrench"       # 修改这里选择要寻找的工具
ENABLE_ROBOT_GRASP = False      # 完成纯视觉坐标验收后才改为 True
# D455 工作台区域：(左, 上, 右, 下)。缩小范围可放大远处的小工具。
D455_ROI = (130, 80, 530, 420)
STABLE_FRAMES = 3               # 连续至少3次有效结果才可能标记为稳定
ROBOT_IP = "192.168.1.35"
HI_SERIAL = "215222074676"     # 手内 D435I（机器人内置相机）
HO_SERIAL = "215122257404"     # 手外 D455（固定相机）
CALIB_PATH = str(SCRIPT_ROOT / "camera_pose.txt")
DEPTH_SCALE_FILE = str(SCRIPT_ROOT / "camera_depth_scale.txt")
CAM_INI = str(SCRIPT_ROOT / "camera_20260906.ini")
CAM2END_PATH = str(SCRIPT_ROOT / "cam2end_20260906.txt")

# 手外 D455 内参（与 camera_pose.txt 标定时所用一致）
HO_FX = 386.471
HO_FY = 386.034
HO_CX = 321.617
HO_CY = 237.200
# 手外内参与标定值容差（像素）
CALIB_TOL = 3.0

TOOL_ORIENTATION = [3.141, 0.0, 0.0]   # 固定朝下 (RX, RY, RZ)
ORIENTATION_SAFE_Z = 0.20     # 姿态归正前TCP至少升到此高度(m)
ORIENTATION_TOL_DEG = 2.0     # 实际姿态与标准姿态的最大允许误差(度)
LIFT_ABOVE = 0.05             # 目标正上方 50mm
CYL_H = 0.03                  # 圆柱高(m)，用于下探深度下界（避免穿底）
GRASP_DEPTH_OFFSET = 0.025    # 低于顶面 z 的下探深度（让手指跨住圆柱中下部）
LIFT_Z_OFFSET = 0.15          # 抓起后抬起高度
HO_Z_OFFSET = 0.026           # 手外 D455 高度基准补偿（标定 z 整体偏低 0.026m）
HI_Z_OFFSET = 0.0             # 手内 z 补偿（若顶面 z 偏低可微调）

# 精定位：多次采样取有效值；若连续 REFINE_RETRY_N 次都失败则立即中止抓取
REFINE_RETRY_N = 5

# 夹爪
GRIP_PORT = "COM10"
GRIP_OPEN_POS = 6000          # 张开
GRIP_CLOSE_POS = 11000        # 闭合（参考仓库值，可用 --gripper-test 校定）
GRIP_SPEED = 50
GRIP_FORCE = 50               # 力矩百分比(≤100)，过低压不扁
GRIP_OPEN_SPEED = 100
GRIP_OPEN_FORCE = 40
GRIP_TORQUE_MIN = 80          # 实时力矩(0x060C)阈值: 低于此且力矩未到达即判空抓

WORKSPACE_LIMITS = [[-0.5, 0.05], [-0.80, -0.45], [-0.2, 0.6]]
GRASP_HOME = [-0.4, -0.025, 0.14981] + TOOL_ORIENTATION


def main():
    args = parse_args()

    # 1. 机器人（手内相机 + 夹爪）
    robot = UR_Robot(
        robot_ip=ROBOT_IP,
        is_use_robot=ENABLE_ROBOT_GRASP,
        is_use_camera=True,
        connect_robot=ENABLE_ROBOT_GRASP,
        cam2end_path=CAM2END_PATH,
        cam_ini_path=CAM_INI,
        camera_serial=HI_SERIAL,
        is_use_gripper=ENABLE_ROBOT_GRASP,
        gripper_port=GRIP_PORT,
    )
    if ENABLE_ROBOT_GRASP:
        print("[OK] 机械臂和夹爪已连接:", ROBOT_IP)
    else:
        print("[安全模式] 仅运行视觉定位，未连接机械臂控制和夹爪。")

    # 2. 手外 D455
    ho_cam = Camera(serial=HO_SERIAL)
    print("[OK] HO(D455) 已连接:", HO_SERIAL)
    K_ho = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    depth_scale = float(np.loadtxt(DEPTH_SCALE_FILE))

    # 3. 自检（可选）：比对实时内参与标定内参
    if args.check_calib or ENABLE_ROBOT_GRASP:
        check_calib(robot, ho_cam)

    class ParamHolder:
        cam_intrinsics = K_ho
        workspace_limits = WORKSPACE_LIMITS

    ho = HandOutEyeCalibration(robot=ParamHolder(), calib_path=CALIB_PATH,
                               cam_depth_scale=depth_scale)
    print("[OK] 手外标定加载完成 (camera_pose.txt, cam->base, 米)")

    detector = SamToolDetector(TEXT_PROMPT)
    ho_filter = TemporalResultFilter(stable_frames=STABLE_FRAMES)
    hi_filter = TemporalResultFilter(stable_frames=STABLE_FRAMES)

    win_ho = "HO(D455)_eye_out"
    win_hi = "HI(D435I)_eye_in"
    cv2.namedWindow(win_ho, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_hi, cv2.WINDOW_NORMAL)

    state = {"ho_base": None, "hi_base": None}

    print("\n操作说明:")
    print("  [安全确认] 示教器活动TCP必须是 TCP_clamp，摆正路径周围必须无遮挡。")
    print("  当前目标: %s（双视场文字识别）。" % TEXT_PROMPT)
    print("  g -> 一键抓取：安全升高并摆正 -> 手外粗定位 -> 手内精定位 -> 下降收爪 -> 抬起")
    print("  t -> 仅测试姿态归正：升高到安全高度 -> 摆正并验证；不会靠近圆柱或抓取")
    print("  o -> 夹爪张开     c -> 夹爪闭合     q -> 退出")

    try:
        while True:
            # ---- 手外 D455：粗定位 ----
            ho_color, ho_depth = ho_cam.get_data()
            raw_ho = detector.detect_roi(ho_color, ho_depth, ho_cam.scale, D455_ROI)
            res_ho, ho_status = ho_filter.update(raw_ho)
            ho_disp = detector.draw(ho_color, res_ho)
            cv2.rectangle(ho_disp, D455_ROI[:2], D455_ROI[2:], (255, 180, 0), 1)
            state["ho_base"] = None
            ho_coord = None
            if res_ho is not None and ho_status == "STABLE":
                pt = hand_out_result_to_base(ho, res_ho)
                if pt is not None:
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2]) + HO_Z_OFFSET
                    state["ho_base"] = (x, y, z)
                    ho_coord = "base [%.3f, %.3f, %.3f]" % (x, y, z)
            draw_header(ho_disp, "D455 GLOBAL", ho_status, ho_coord)
            cv2.imshow(win_ho, ho_disp)

            # ---- 手内 D435I：精定位预览 ----
            hi_color, hi_depth = robot.get_camera_data()
            hi_disp = hi_color.copy() if hi_color is not None else np.zeros((480, 640, 3), np.uint8)
            state["hi_base"] = None
            hi_status = "SEARCHING"
            hi_coord = None
            if hi_color is not None:
                raw_hi = detector.detect(hi_color, hi_depth, robot.camera.scale)
                res_hi, hi_status = hi_filter.update(raw_hi)
                hi_disp = detector.draw(hi_color, res_hi)
                if res_hi is not None and res_hi["z_mm"] is not None and hi_status == "STABLE":
                    if ENABLE_ROBOT_GRASP:
                        base_mm, base_m = robot.pixel_to_base(*res_hi["center"], res_hi["z_mm"])
                        x, y, z = float(base_m[0]), float(base_m[1]), float(base_m[2]) + HI_Z_OFFSET
                        state["hi_base"] = (x, y, z)
                        coord_name = "HI base"
                    else:
                        camera_mm = robot.pixel_to_camera(*res_hi["center"], res_hi["z_mm"])
                        x, y, z = (camera_mm / 1000.0).tolist()
                        coord_name = "HI camera"
                    hi_coord = "%s [%.3f, %.3f, %.3f]" % (coord_name, x, y, z)
            draw_header(hi_disp, "D435I WRIST", hi_status, hi_coord)
            cv2.imshow(win_hi, hi_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('o'):
                if ENABLE_ROBOT_GRASP:
                    robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
                    print("[夹爪] 张开 pos=%d" % GRIP_OPEN_POS)
                else:
                    print("[安全锁定] 纯视觉模式不控制夹爪。")
            elif key == ord('c'):
                if ENABLE_ROBOT_GRASP:
                    robot.grip(GRIP_CLOSE_POS, GRIP_SPEED, GRIP_FORCE)
                    print("[夹爪] 闭合 pos=%d" % GRIP_CLOSE_POS)
                else:
                    print("[安全锁定] 纯视觉模式不控制夹爪。")
            elif key == ord('t'):
                if ENABLE_ROBOT_GRASP:
                    print("[姿态测试] 仅执行安全升高和末端摆正，不执行抓取")
                    normalize_tool_pose(robot)
                else:
                    print("[安全锁定] 纯视觉模式不发送机械臂运动命令。")
            elif key == ord('g'):
                if ENABLE_ROBOT_GRASP:
                    do_grasp(robot, detector, state)
                else:
                    print("[安全锁定] 当前仅验证识别与坐标。确认无误后将 ENABLE_ROBOT_GRASP 改为 True。")
    finally:
        if getattr(robot, "camera", None) is not None:
            robot.camera.stop()
        ho_cam.stop()
        cv2.destroyAllWindows()
        print("退出。")


def draw_header(image, camera_name, status, coordinate=None):
    """Draw non-overlapping camera, tracking status and coordinate lines."""
    colors = {
        "SEARCHING": (0, 0, 255),
        "TRACKING": (0, 200, 255),
        "STABLE": (0, 255, 0),
    }
    cv2.rectangle(image, (0, 0), (image.shape[1], 72), (20, 20, 20), -1)
    cv2.putText(image, "%s  %s" % (camera_name, status), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, colors.get(status, (255, 255, 255)),
                2, cv2.LINE_AA)
    detail = coordinate or "coordinate unavailable until STABLE"
    cv2.putText(image, detail, (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (230, 230, 230), 1, cv2.LINE_AA)


def check_calib(robot, ho_cam):
    """比对实时内参与标定内参，超阈值则抛错拒绝执行。"""
    ok = True

    # ---- 手外 D455 ----
    live = ho_cam.intrinsics
    exp = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    dho = max(abs(live[0, 0] - exp[0, 0]), abs(live[1, 1] - exp[1, 1]),
              abs(live[0, 2] - exp[0, 2]), abs(live[1, 2] - exp[1, 2]))
    print("手外D455 实时 fx=%.3f fy=%.3f cx=%.3f cy=%.3f | 期望=%.3f/%.3f/%.3f/%.3f | Δmax=%.3f px"
          % (live[0, 0], live[1, 1], live[0, 2], live[1, 2],
             exp[0, 0], exp[1, 1], exp[0, 2], exp[1, 2], dho))
    if dho > CALIB_TOL:
        print("[自检失败] 手外D455 内参与标定值不符(%.3f>%.1f)！"
              % (dho, CALIB_TOL))
        ok = False

    # ---- 手内 D435I ----
    calib = load_camera_ini(CAM_INI)
    if calib is None:
        print("[警告] 未找到/无法解析 %s，手内内参无法校验" % CAM_INI)
    else:
        K_ini, _ = calib
        live_hi = robot.camera.intrinsics
        dhi = max(abs(live_hi[0, 0] - K_ini[0, 0]), abs(live_hi[1, 1] - K_ini[1, 1]),
                  abs(live_hi[0, 2] - K_ini[0, 2]), abs(live_hi[1, 2] - K_ini[1, 2]))
        print("手内D435I 实时 fx=%.3f fy=%.3f cx=%.3f cy=%.3f | 期望=%.3f/%.3f/%.3f/%.3f | Δmax=%.3f px"
              % (live_hi[0, 0], live_hi[1, 1], live_hi[0, 2], live_hi[1, 2],
                 K_ini[0, 0], K_ini[1, 1], K_ini[0, 2], K_ini[1, 2], dhi))
        if dhi > CALIB_TOL:
            print("[自检失败] 手内D435I 内参与标定值不符(%.3f>%.1f)！" % (dhi, CALIB_TOL))
            ok = False

    if not ok:
        raise RuntimeError("内参自检未通过，请检查相机/标定。不要继续执行抓取。")


def hand_out_result_to_base(ho, result):
    """Use robust mask depth instead of a possibly empty center depth pixel."""
    if result is None or result["z_mm"] is None:
        return None
    u, v = result["center"]
    z = float(result["z_mm"]) / 1000.0
    k = ho.cam_intrinsics
    point = np.array([(u - k[0, 2]) * z / k[0, 0],
                      (v - k[1, 2]) * z / k[1, 1], z, 1.0])
    return (ho.camera2robot_pose @ point)[:3]


def _read_valid_tcp_pose(robot):
    """读取并校验UR返回的[x,y,z,rx,ry,rz] TCP位姿。"""
    pose = np.asarray(robot.get_actual_tcp_pose(), dtype=np.float64).reshape(-1)
    if pose.size != 6 or not np.all(np.isfinite(pose)):
        raise ValueError("无效TCP位姿: %s" % pose)
    return pose


def _orientation_error_deg(actual_rvec, target_rvec):
    """用旋转矩阵夹角计算姿态误差，避免旋转向量等价表示造成误判。"""
    actual_R, _ = cv2.Rodrigues(np.asarray(actual_rvec, dtype=np.float64).reshape(3, 1))
    target_R, _ = cv2.Rodrigues(np.asarray(target_rvec, dtype=np.float64).reshape(3, 1))
    delta_R = target_R @ actual_R.T
    cos_angle = np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def verify_tool_orientation(robot, stage):
    """检查末端是否到达标准朝下姿态；读取或误差异常时返回False。"""
    try:
        actual = _read_valid_tcp_pose(robot)
        error_deg = _orientation_error_deg(actual[3:6], TOOL_ORIENTATION)
        print("[姿态验证:%s] 实际TCP=%s" %
              (stage, ["%.4f" % v for v in actual]))
        print("[姿态验证:%s] 与标准朝下姿态误差=%.3f°（允许≤%.1f°）" %
              (stage, error_deg, ORIENTATION_TOL_DEG))
        if error_deg > ORIENTATION_TOL_DEG:
            print("[安全中止] 末端姿态未摆正，禁止继续靠近或下降")
            return False
        return True
    except Exception as exc:
        print("[安全中止] 无法验证末端姿态：%s" % exc)
        return False


def normalize_tool_pose(robot):
    """先垂直升到安全高度，再原地摆正末端；失败时停止后续流程。"""
    try:
        current = _read_valid_tcp_pose(robot)
        print("[姿态归正] 当前TCP=%s" % (["%.4f" % v for v in current],))

        lift_target = current.copy()
        lift_target[2] = max(float(current[2]), ORIENTATION_SAFE_Z)
        print("[姿态归正] 安全抬升目标=%s" %
              (["%.4f" % v for v in lift_target],))
        if lift_target[2] > current[2] + 0.001:
            robot.moveL(lift_target.tolist(), speed=0.05, acceleration=0.05)
        else:
            print("[姿态归正] 当前TCP已不低于安全高度，无需向下或重复抬升")

        after_lift = _read_valid_tcp_pose(robot)
        straighten_target = after_lift.copy()
        straighten_target[3:6] = TOOL_ORIENTATION
        print("[姿态归正] 标准姿态目标=%s" %
              (["%.4f" % v for v in straighten_target],))
        robot.moveL(straighten_target.tolist(), speed=0.05, acceleration=0.05)

        if not verify_tool_orientation(robot, "摆正后"):
            return False
        print("[姿态归正] 完成，可以继续执行粗定位")
        return True
    except Exception as exc:
        print("[安全中止] 姿态归正失败：%s" % exc)
        try:
            if robot.rtde_c is not None and hasattr(robot.rtde_c, "stopL"):
                robot.rtde_c.stopL(1.0)
        except Exception as stop_exc:
            print("[提示] 停止直线运动命令执行失败，请使用示教器或急停检查：%s" % stop_exc)
        return False


def do_grasp(robot, detector, state):
    """一键全流程：① 粗定位 -> ② 精定位 -> ③ 下降收爪抬起；精定位失败则安全中止。"""
    coarse = state["ho_base"] or state["hi_base"]
    if coarse is None:
        print("[提示] 未检测到目标工具，无法抓取")
        return

    x, y, z = coarse
    # ③ 工作空间钳制：x/y 同样限制，防止目标点在可达范围外
    clamp = lambda v, lim: max(lim[0], min(v, lim[1]))
    x = clamp(x, WORKSPACE_LIMITS[0])
    y = clamp(y, WORKSPACE_LIMITS[1])
    z = clamp(z, WORKSPACE_LIMITS[2])
    try:
        # ① 粗定位：手外基座坐标 -> 移到目标上方
        print("[①粗定位] 手外 base = [%.3f, %.3f, %.3f]" % (x, y, z))
        robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
        if not normalize_tool_pose(robot):
            print("[安全中止] 姿态归正未通过，本次抓取结束；夹爪保持打开")
            return
        above = [x, y, z + LIFT_ABOVE] + TOOL_ORIENTATION
        print("[移动] 粗定位上方 moveL %s" % (["%.3f" % v for v in above],))
        robot.moveL(above, speed=0.05, acceleration=0.05)
        if not verify_tool_orientation(robot, "粗定位上方"):
            print("[安全中止] 到达粗定位上方后姿态异常，不执行手内精定位或下降")
            return
        time.sleep(1)

        # ② 精定位：手内多帧重试；失败则立即中止，禁止使用粗定位坐标下降
        refined = refine_locate(robot, detector, REFINE_RETRY_N)
        if refined is not None:
            x, y, z = refined
            x = clamp(x, WORKSPACE_LIMITS[0])
            y = clamp(y, WORKSPACE_LIMITS[1])
            z = clamp(z, WORKSPACE_LIMITS[2])
            print("[②精定位] 手内 base = [%.3f, %.3f, %.3f]" % (x, y, z))
        else:
            print("[安全中止] 手内精定位连续 %d 次失败，禁止使用粗定位坐标下降抓取" % REFINE_RETRY_N)
            print("[安全中止] 机械臂保持在粗定位上方，夹爪保持打开；请检查目标和手内相机后重新按 g")
            return

        # ③ 精对齐 -> 下降 -> 收爪 -> 抬起
        above = [x, y, z + LIFT_ABOVE] + TOOL_ORIENTATION
        print("[移动] 精定位上方 moveL %s" % (["%.3f" % v for v in above],))
        robot.moveL(above, speed=0.05, acceleration=0.05)

        clamp = lambda v, lim: max(lim[0], min(v, lim[1]))
        z_low = clamp(z - GRASP_DEPTH_OFFSET, WORKSPACE_LIMITS[2])
        floor_z = z - CYL_H               # 圆柱底面的 z，下探不得低于此
        if z_low < floor_z:
            print("[安全] 下探 z=%.3f 低于圆柱底面 z=%.3f，抬至底面高度" % (z_low, floor_z))
            z_low = floor_z
        grasp_pt = [x, y, z_low] + TOOL_ORIENTATION
        print("[移动] 下降抓取 moveL %s" % (["%.3f" % v for v in grasp_pt],))
        robot.moveL(grasp_pt, speed=0.05, acceleration=0.05)

        print("[夹爪] 收爪抓取 pos=%d" % GRIP_CLOSE_POS)
        robot.grip(GRIP_CLOSE_POS, GRIP_SPEED, GRIP_FORCE)
        torq_reached = robot.read_torque_reached()
        torq_cur = robot.read_torque_current()
        print("[夹爪] 力矩到达=%d  实时力矩=%d" % (torq_reached, torq_cur))
        if not (torq_reached == 1 or torq_cur >= GRIP_TORQUE_MIN):
            print("[抓取失败] 力矩未到达且实时力矩过低(疑似空抓)，松开并中止，不抬升")
            robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
            return
        time.sleep(1)

        lift = [x, y, z + LIFT_Z_OFFSET] + TOOL_ORIENTATION
        print("[移动] 抬起 moveL %s" % (["%.3f" % v for v in lift],))
        robot.moveL(lift, speed=0.05, acceleration=0.05)

        print("抓取完成（已抓起并抬起，未放置）。按 o 张开可放下，q 退出。")
    except Exception as e:
        print("[异常] 抓取流程出错：%s" % e)
        safe_return(robot)


def refine_locate(robot, detector, n_repeat):
    """手内精定位，重复 n_repeat 次取第一个有效值；全失败返回 None。"""
    for i in range(n_repeat):
        hi_color, hi_depth = robot.get_camera_data()
        if hi_color is not None:
            res = detector.detect(hi_color, hi_depth, robot.camera.scale)
            if res is not None and res["z_mm"] is not None:
                base_mm, base_m = robot.pixel_to_base(*res["center"], res["z_mm"])
                return (float(base_m[0]), float(base_m[1]),
                        float(base_m[2]) + HI_Z_OFFSET)
        time.sleep(0.3)
    return None


def safe_return(robot):
    """异常时回 home 并张开爪。"""
    try:
        robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
        robot.moveL(GRASP_HOME, speed=0.05, acceleration=0.05)
    except Exception as e:
        print("[提示] 复位失败，请手动检查机器人：%s" % e)


def parse_args():
    p = argparse.ArgumentParser(description="文字指定工具抓取（手外粗定位->手内精定位->抓取）")
    p.add_argument("--gripper-test", action="store_true",
                   help="仅测试夹爪：交互式发送 position 并回显，用于定开/合值")
    p.add_argument("--check-calib", action="store_true",
                   help="启动时读实时内参并比对标定内参，超阈值则拒绝执行")
    return p.parse_args()


def gripper_test():
    """交互式夹爪自检：输入 position -> 控制并回显当前 POS。"""
    robot = UR_Robot(robot_ip=ROBOT_IP, is_use_robot=True, is_use_camera=False,
                     is_use_gripper=True, gripper_port=GRIP_PORT)
    print("夹爪自检：输入 position（0~65535）控制，观察爪的开合。")
    print("  数值越小越张开、越大越闭合。q 退出。")
    while True:
        s = input("position(回车看当前值) / q 退出 > ").strip()
        if not s:
            print("当前POS =", robot.read_position())
            continue
        if s.lower() == 'q':
            break
        try:
            pos = int(s)
        except ValueError:
            print("请输入整数")
            continue
        robot.grip(pos, 100, 40)
        print("已发 position=%d，回读 POS=%d" % (pos, robot.read_position()))
    print("结束。把张开/闭合对应的 position 填到脚本的 GRIP_OPEN_POS/GRIP_CLOSE_POS。")


if __name__ == "__main__":
    args = parse_args()
    if args.gripper_test:
        gripper_test()
    else:
        main()
