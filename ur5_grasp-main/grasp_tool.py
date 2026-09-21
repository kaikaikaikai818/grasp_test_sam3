#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文字指定工具抓取（D455 全局粗定位 -> D435i 腕部精定位 -> 自适应夹持，含夹爪）

一键抓取（按 a）：
  ① D455 粗定位到目标上方（仅安全高度移动，不下降）
  ② 等 D435i 稳定后自动无接触下降（D 终点 = 夹持点上方 40mm）
  ③ 自动下降到夹持高度 -> 低力闭爪 -> 力矩确认 -> 抬升并停住
  任一步超时或不通过都会停在安全位置，绝不盲降。

手动按键：
  a 一键抓取（D455 粗定位 + 自动下降/夹持）
  p 仅双相机观察点   d 无接触下降   r 夹持并抬升
  t 仅升高并摆正     o 夹爪张开     c 夹爪闭合     q 退出
自检：
  --check-calib  启动时读两台相机实时内参，与标定内参比对，超阈值拒绝执行
  --gripper-test 交互式校定夹爪开/合 position
"""
import argparse
import json
import os
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
warnings.filterwarnings("ignore")
import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot, load_camera_ini
from bsp.robot_bsp.read_only_robot_state import ReadOnlyRobotState
from bsp.robot_bsp.fixed_placement import load_placement, place_at_fixed_point
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.camera_bsp.hand_out_eye_calibration import HandOutEyeCalibration
from bsp.camera_bsp.sam_tool_detect import SamToolDetector, TemporalResultFilter
from bsp.camera_bsp.camera_alignment import apply_alignment, load_alignment
from bsp.camera_bsp.camera_profile import max_intrinsics_delta
from bsp.camera_bsp.tool_names import known_category, normalize_prompt
from bsp.camera_bsp.tool_grasp_candidates import propose_grasp_region
from bsp.camera_bsp.tool_profiles import load_tool_profile
from bsp.camera_bsp.planar_orientation import (axial_difference_deg,
                                                overhead_orientation,
                                                principal_axis_base)
from bsp.camera_bsp.screwdriver_grasp import (estimate_handle_thickness,
                                               find_screwdriver_handle,
                                               fit_horizontal_support_plane,
                                               load_calibration, plan_grasp_tcp,
                                               result_at_handle)

# -------------------------- 配置 --------------------------
SCRIPT_ROOT = Path(__file__).resolve().parent
TEXT_PROMPT = "a screwdriver"       # 修改这里选择要寻找的工具
ENABLE_ROBOT_GRASP = False      # 完成纯视觉坐标验收后才改为 True
# 螺丝刀实际夹持：仅在完成深度标定、P 和 D 后按 R 才会执行；默认关闭。
ENABLE_SCREWDRIVER_GRASP = True
ENABLE_ROBOT_STATE_READ = True  # 只读TCP位姿；不会创建机械臂控制接口
# 安全观察点测试：仅按 P 后移动到目标上方；不初始化夹爪、不下降、不抓取。
# 它与 ENABLE_ROBOT_GRASP 互斥，默认关闭。
ENABLE_SAFE_APPROACH_TEST = True
# 无接触下降测试：只在观察点确认后按 D 低速下降到工具上方；默认关闭。
ENABLE_SAFE_DESCENT_TEST = True
# D455 工作台区域：(左, 上, 右, 下)。缩小范围可放大远处的小工具。
D455_ROI = (170, 95, 500, 370)
STABLE_FRAMES = 3               # 连续至少3次有效结果才可能标记为稳定
# 视觉验收门槛。这里只决定坐标是否值得记录，不会授权机械臂运动。
D455_MIN_SCORE = 0.35
D435I_MIN_SCORE = 0.45
MIN_BOX_SIDE_PX = 12
MIN_VALID_DEPTH_POINTS = 80
MEASUREMENT_LOG_INTERVAL_S = 2.0
TARGET_ASSOCIATION_MAX_DISTANCE_M = 0.10
APPROACH_HEIGHT_M = 0.15
# P 键运动的独立安全门槛：必须连续三帧双相机一致到 15mm 内。
SAFE_APPROACH_ASSOCIATION_MAX_DISTANCE_M = 0.015
SAFE_APPROACH_CONFIRM_FRAMES = 3
SAFE_TRAVEL_Z_M = 0.25
SAFE_APPROACH_SPEED = 0.03
SAFE_APPROACH_ACCELERATION = 0.03
# 观察点后的抓取预览：只显示计划，不会产生任何运动命令。
GRASP_PREVIEW_HEIGHT_M = 0.10
GRASP_PREVIEW_SURFACE_CLEARANCE_M = 0.025
# 第一次实体下降保留更大的 40mm 间隙，不使用虚拟预览的 25mm 终点。
SAFE_DESCENT_CLEARANCE_M = 0.040
SAFE_DESCENT_CONFIRM_FRAMES = 3
SAFE_DESCENT_SPEED = 0.01
SAFE_DESCENT_ACCELERATION = 0.01
SCREWDRIVER_GRASP_SPEED = 0.01
SCREWDRIVER_GRASP_FORCE = 30
SCREWDRIVER_TEST_LIFT_M = 0.050
SCREWDRIVER_TARGET_SHIFT_MAX_M = 0.030
SCREWDRIVER_R_START_TOL_M = 0.015
SUPPORT_PLANE_SHIFT_MAX_M = 0.008
VALIDATION_DIR = SCRIPT_ROOT.parent / "outputs" / "validation"
POSITION_LABELS = {
    ord("1"): "center",
    ord("2"): "left",
    ord("3"): "right",
    ord("4"): "top",
    ord("5"): "bottom",
}
ROBOT_IP = "192.168.1.35"
HI_SERIAL = "215222074676"     # 手内 D435I（机器人内置相机）
HO_SERIAL = "215122257404"     # 手外 D455（固定相机）
CALIB_PATH = str(SCRIPT_ROOT / "camera_pose.txt")
DEPTH_SCALE_FILE = str(SCRIPT_ROOT / "camera_depth_scale.txt")
CAM_INI = str(SCRIPT_ROOT / "camera_20260906.ini")
CAM2END_PATH = str(SCRIPT_ROOT / "cam2end_20260906.txt")
CAMERA_ALIGNMENT_PATH = SCRIPT_ROOT / "camera_alignment.json"
SCREWDRIVER_CALIBRATION_PATH = SCRIPT_ROOT / "grasp_surface_calibration.json"

# 手外 D455 内参（与 camera_pose.txt 标定时所用一致）
HO_FX = 386.471
HO_FY = 386.034
HO_CX = 321.617
HO_CY = 237.200
# 相机内参与标定值容差（像素）。D435i 的 8px 上限仅适用于当前已验证的
# 640x480 流配置；D455 仍保持严格阈值。
D455_CALIB_TOL = 3.0
D435I_CALIB_TOL = 8.0
D435I_VERIFIED_WARNING_TOL = 3.0

TOOL_ORIENTATION = [3.141, 0.0, 0.0]   # 固定朝下 (RX, RY, RZ)
ORIENTATION_SAFE_Z = 0.20     # 姿态归正前TCP至少升到此高度(m)
ORIENTATION_TOL_DEG = 2.0     # 实际姿态与标准姿态的最大允许误差(度)
HO_Z_OFFSET = 0.026           # 手外 D455 高度基准补偿（标定 z 整体偏低 0.026m）
HI_Z_OFFSET = 0.0             # 手内 z 补偿（若顶面 z 偏低可微调）

# 一键抓取：a 粗定位后自动完成下降与夹持；等待相机稳定的超时(秒)
ENABLE_ONE_KEY_GRASP = True
AUTO_GRASP_TIMEOUT_S = 8.0
ANGLE_GRASP_TIMEOUT_S = 15.0
ANGLE_STABLE_DEG = 5.0

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


def main():
    global TEXT_PROMPT, ENABLE_ROBOT_GRASP, ENABLE_SAFE_APPROACH_TEST
    global ENABLE_SAFE_DESCENT_TEST, ENABLE_SCREWDRIVER_GRASP, ENABLE_ONE_KEY_GRASP
    args = parse_args()
    if args.prompt:
        TEXT_PROMPT = args.prompt
    TEXT_PROMPT = normalize_prompt(TEXT_PROMPT)
    tool_category = known_category(TEXT_PROMPT)
    # Every run starts locked. The requested stage explicitly opens only the
    # capabilities needed for that validation step.
    ENABLE_ROBOT_GRASP = False
    ENABLE_SAFE_APPROACH_TEST = args.stage in ("observe", "rotate", "descent", "grasp", "place")
    ENABLE_SAFE_DESCENT_TEST = args.stage in ("descent", "grasp", "place")
    ENABLE_SCREWDRIVER_GRASP = args.stage in ("grasp", "place")
    ENABLE_ONE_KEY_GRASP = bool(args.auto and args.stage in ("grasp", "place"))
    args.enable_angle_rotation = args.stage in ("rotate", "descent", "grasp", "place")
    args.place_after_grasp = args.stage == "place"
    args.vision_only = args.stage == "vision"
    profile = None
    if tool_category != "screwdriver" and args.stage in ("rotate", "descent", "grasp", "place"):
        profile = load_tool_profile(args.profile_config, tool_category, WORKSPACE_LIMITS)
    if args.enable_angle_rotation and tool_category != "screwdriver" and profile is None:
        raise ValueError("non-screwdriver rotation requires an approved grasp profile")
    if profile and profile["requires_angle"] and not args.enable_angle_rotation:
        raise ValueError("this tool profile requires --stage rotate or a later stage")
    if args.place_after_grasp and (tool_category != "screwdriver" and profile is None):
        raise ValueError("fixed placement requires an enabled grasp path")
    placement = (load_placement(args.place_config, WORKSPACE_LIMITS)
                 if args.place_after_grasp else None)
    if ENABLE_ROBOT_GRASP and ENABLE_SAFE_APPROACH_TEST:
        raise RuntimeError("ENABLE_ROBOT_GRASP 与 ENABLE_SAFE_APPROACH_TEST 不能同时开启")
    if ENABLE_SAFE_DESCENT_TEST and not ENABLE_SAFE_APPROACH_TEST:
        raise RuntimeError("ENABLE_SAFE_DESCENT_TEST 需要先开启 ENABLE_SAFE_APPROACH_TEST")
    robot_control_enabled = (ENABLE_ROBOT_GRASP or ENABLE_SAFE_APPROACH_TEST
                             or ENABLE_SCREWDRIVER_GRASP)
    gripper_enabled = ENABLE_ROBOT_GRASP or ENABLE_SCREWDRIVER_GRASP

    # 1. 机器人（手内相机 + 夹爪）
    robot = UR_Robot(
        robot_ip=ROBOT_IP,
        is_use_robot=robot_control_enabled,
        is_use_camera=True,
        connect_robot=robot_control_enabled,
        cam2end_path=CAM2END_PATH,
        cam_ini_path=CAM_INI,
        camera_serial=HI_SERIAL,
        is_use_gripper=gripper_enabled,
        gripper_port=GRIP_PORT,
    )
    robot_state = ReadOnlyRobotState(
        ROBOT_IP, enabled=ENABLE_ROBOT_STATE_READ and not robot_control_enabled)
    if ENABLE_ROBOT_GRASP:
        print("[OK] 机械臂和夹爪已连接:", ROBOT_IP)
    elif ENABLE_SAFE_APPROACH_TEST:
        print("[观察点模式] 机械臂控制已连接；夹爪%s初始化。" %
              ("已" if gripper_enabled else "未"))
    else:
        print("[安全模式] 仅运行视觉定位，未连接机械臂控制和夹爪。")
        if robot_state.available:
            print("[只读模式] 已连接机械臂状态接口，只读取TCP位姿。")
        else:
            print("[只读模式] TCP位姿不可用，D435i将只显示相机坐标。")

    # 2. 手外 D455
    ho_cam = Camera(serial=HO_SERIAL)
    print("[OK] HO(D455) 已连接:", HO_SERIAL)
    K_ho = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    depth_scale = float(np.loadtxt(DEPTH_SCALE_FILE))

    # 3. 自检（可选）：比对实时内参与标定内参
    if args.check_calib or ENABLE_ROBOT_GRASP or ENABLE_SCREWDRIVER_GRASP:
        check_calib(robot, ho_cam)

    class ParamHolder:
        cam_intrinsics = K_ho
        workspace_limits = WORKSPACE_LIMITS

    ho = HandOutEyeCalibration(robot=ParamHolder(), calib_path=CALIB_PATH,
                               cam_depth_scale=depth_scale)
    print("[OK] 手外标定加载完成 (camera_pose.txt, cam->base, 米)")
    try:
        camera_alignment = load_alignment(CAMERA_ALIGNMENT_PATH)
    except Exception as exc:
        camera_alignment = None
        print("[校正文件无效] 使用D455原始坐标：%s" % exc)
    if camera_alignment is None:
        print("[双相机校正] 未加载，目标关联使用D455原始坐标。")
    else:
        print("[双相机校正] 已加载:", CAMERA_ALIGNMENT_PATH)

    detector = SamToolDetector(TEXT_PROMPT, checkpoint=args.checkpoint,
                               backend=args.backend)
    ho_filter = TemporalResultFilter(stable_frames=STABLE_FRAMES)
    hi_filter = TemporalResultFilter(stable_frames=STABLE_FRAMES)
    angle_history = deque(maxlen=3)
    angle_rotation_completed = False

    win_ho = "HO(D455)_eye_out"
    win_hi = "HI(D435I)_eye_in"
    cv2.namedWindow(win_ho, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_hi, cv2.WINDOW_NORMAL)

    state = {"ho_base": None, "ho_base_aligned": None,
             "hi_base": None, "association": None}
    approach_ready_streak = 0
    approach_destination = None
    coarse_ready_streak = 0
    coarse_destination = None
    coarse_reason = None
    auto_grasp_state = "idle"
    auto_grasp_deadline = 0.0
    d455_detection_frozen = False
    observation_active = False
    descent_ready_streak = 0
    locked_grasp_preview = None
    locked_screwdriver_handle = None
    safe_descent_completed = False
    last_log_time = 0.0
    active_position = None
    position_counts = {label: 0 for label in POSITION_LABELS.values()}
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_log = VALIDATION_DIR / ("measurements_%s.jsonl" % session_id)

    print("\n操作说明:")
    print("  [安全确认] 示教器活动TCP必须是 TCP_clamp，摆正路径周围必须无遮挡。")
    print("  当前目标: %s（双视场文字识别）。" % TEXT_PROMPT)
    screwdriver_calibration_error = None
    try:
        screwdriver_calibration = load_calibration(SCREWDRIVER_CALIBRATION_PATH)
    except Exception as exc:
        screwdriver_calibration = None
        screwdriver_calibration_error = str(exc)
    if TEXT_PROMPT.strip().lower() == "a screwdriver":
        if screwdriver_calibration is None:
            print("  [螺丝刀抓取锁定] 尚未完成深度标定：运行 calibrate_screwdriver_grasp.py plane/contact。")
            if screwdriver_calibration_error:
                print("      标定文件读取失败：%s" % screwdriver_calibration_error)
        else:
            print("  [螺丝刀标定] 已加载：支撑面 z=%.4fm。R 键仍需 ENABLE_SCREWDRIVER_GRASP=True。" %
                  screwdriver_calibration["support_plane_z_m"])
    if ENABLE_SAFE_APPROACH_TEST:
        print("  P -> 仅到安全观察点：先升至 %.0fmm，再摆正并水平移动到目标上方；不会下降或控制夹爪"
              % (SAFE_TRAVEL_Z_M * 1000.0))
        print("      仅当双相机连续 %d 帧一致且误差≤%.0fmm 时允许执行。"
              % (SAFE_APPROACH_CONFIRM_FRAMES,
                 SAFE_APPROACH_ASSOCIATION_MAX_DISTANCE_M * 1000.0))
        print("  a -> D455单相机粗定位：腕部相机看不到工具时，仅用D455对齐坐标移动到目标上方。")
        if ENABLE_ONE_KEY_GRASP:
            print("       随后自动完成无接触下降与夹持（一键抓取，超时 %.0f 秒即停在安全位置）。"
                  % AUTO_GRASP_TIMEOUT_S)
    if ENABLE_SAFE_DESCENT_TEST:
        print("  D -> 无接触下降测试：仅在观察点、D435i连续 %d 帧稳定后，低速停在目标上方 %.0fmm；不控制夹爪"
              % (SAFE_DESCENT_CONFIRM_FRAMES, SAFE_DESCENT_CLEARANCE_M * 1000.0))
    print("  R -> 已审核工具低力夹持并抬升 %.0fmm（需对应工具标定）"
          % (SCREWDRIVER_TEST_LIFT_M * 1000.0))
    print("  t -> 仅测试姿态归正：升高到安全高度 -> 摆正并验证；不会靠近工具或抓取")
    print("  o -> 夹爪张开     c -> 夹爪闭合     q -> 退出")
    print("  坐标验收: 1中心  2左侧  3右侧  4上方  5下方")
    print("  本次测量文件:", validation_log)

    try:
        while True:
            # ---- 手外 D455：粗定位 ----
            ho_color, ho_depth = ho_cam.get_data()
            state["ho_base"] = None
            state["ho_base_aligned"] = None
            ho_coord = None
            if d455_detection_frozen:
                # Once the arm moves, its gripper may enter D455's view and is
                # visually similar to a tool.  D455 is no longer used after P.
                res_ho = None
                ho_status = "FROZEN"
                ho_gate = {"passed": False, "reasons": ["paused after P movement"]}
                ho_coord = "D455 detection paused after P"
                ho_disp = ho_color.copy()
            else:
                raw_ho = detector.detect_roi(ho_color, ho_depth, ho_cam.scale, D455_ROI)
                res_ho, ho_status = ho_filter.update(raw_ho)
                ho_handle = None
                ho_handle_reason = None
                if (TEXT_PROMPT.strip().lower() == "a screwdriver" and res_ho is not None
                        and ho_status == "STABLE"):
                    ho_handle, ho_handle_reason = find_screwdriver_handle(
                        res_ho, ho_depth, ho_cam.scale)
                    if ho_handle is not None:
                        res_ho = result_at_handle(res_ho, ho_handle)
                elif tool_category is not None and res_ho is not None and ho_status == "STABLE":
                    candidate, ho_handle_reason = propose_grasp_region(
                        res_ho["mask"], ho_depth, ho_cam.scale, tool_category)
                    if candidate is not None:
                        res_ho = dict(res_ho)
                        res_ho["center"] = candidate.center_px
                        res_ho["z_mm"] = candidate.depth_m * 1000.0
                ho_disp = detector.draw(ho_color, res_ho)
                ho_gate = gate_detection(res_ho, ho_status, "D455")
                if ho_handle_reason is not None and ho_handle is None:
                    ho_gate = {"passed": False, "reasons": [ho_handle_reason]}
                if (res_ho is not None and ho_status == "STABLE"
                        and ho_handle_reason is None):
                    pt = hand_out_result_to_base(ho, res_ho)
                    if pt is not None:
                        x, y, z = float(pt[0]), float(pt[1]), float(pt[2]) + HO_Z_OFFSET
                        state["ho_base"] = (x, y, z)
                        state["ho_base_aligned"] = (
                            apply_alignment(state["ho_base"], camera_alignment)
                            if camera_alignment is not None else state["ho_base"])
                        ho_gate = gate_detection(res_ho, ho_status, "D455", state["ho_base"])
                        ax, ay, az = state["ho_base_aligned"]
                        coord_kind = "aligned" if camera_alignment is not None else "raw"
                        ho_coord = "%s  %s [%.3f, %.3f, %.3f]" % (
                            "PASS" if ho_gate["passed"] else "REJECT",
                            coord_kind, ax, ay, az)
            cv2.rectangle(ho_disp, D455_ROI[:2], D455_ROI[2:], (255, 180, 0), 1)
            position_text = active_position.upper() if active_position else "PRESS 1-5"
            sample_count = position_counts.get(active_position, 0)

            # ---- 手内 D435I：精定位预览 ----
            hi_color, hi_depth = robot.get_camera_data()
            hi_disp = hi_color.copy() if hi_color is not None else np.zeros((480, 640, 3), np.uint8)
            state["hi_base"] = None
            hi_status = "SEARCHING"
            hi_coord = None
            res_hi = None
            hi_camera = None
            tcp_pose = None
            hi_handle = None
            hi_handle_reason = None
            current_support_plane = None
            support_plane_reason = None
            current_axis = None
            current_orientation = None
            hi_gate = gate_detection(None, hi_status, "D435I")
            if hi_color is not None:
                raw_hi = detector.detect(hi_color, hi_depth, robot.camera.scale)
                res_hi, hi_status = hi_filter.update(raw_hi)
                if (TEXT_PROMPT.strip().lower() == "a screwdriver" and res_hi is not None
                        and hi_status == "STABLE"):
                    hi_handle, hi_handle_reason = find_screwdriver_handle(
                        res_hi, hi_depth, robot.camera.scale)
                    if hi_handle is not None:
                        res_hi = result_at_handle(res_hi, hi_handle)
                elif tool_category is not None and res_hi is not None and hi_status == "STABLE":
                    candidate, hi_handle_reason = propose_grasp_region(
                        res_hi["mask"], hi_depth, robot.camera.scale, tool_category)
                    if candidate is not None:
                        res_hi = dict(res_hi)
                        res_hi["center"] = candidate.center_px
                        res_hi["z_mm"] = candidate.depth_m * 1000.0
                hi_disp = detector.draw(hi_color, res_hi)
                hi_gate = gate_detection(res_hi, hi_status, "D435I")
                if hi_handle_reason is not None and hi_handle is None:
                    hi_gate = {"passed": False, "reasons": [hi_handle_reason]}
                if (res_hi is not None and res_hi["z_mm"] is not None
                        and hi_status == "STABLE" and hi_handle_reason is None):
                    camera_mm = robot.pixel_to_camera(*res_hi["center"], res_hi["z_mm"])
                    hi_camera = tuple((camera_mm / 1000.0).tolist())
                    # Full grasp and safe-observation mode both own the robot
                    # control connection, so either mode can read the live TCP.
                    if robot_control_enabled or robot_state.available:
                        try:
                            tcp_source = robot if robot_control_enabled else robot_state
                            tcp_pose = _read_valid_tcp_pose(tcp_source)
                            _, base_m = robot.camera_to_base(camera_mm, tcp_pose=tcp_pose)
                            x, y, z = [float(value) for value in base_m]
                            z += HI_Z_OFFSET
                            state["hi_base"] = (x, y, z)
                            hi_gate = gate_detection(
                                res_hi, hi_status, "D435I", state["hi_base"])
                            def hi_pixel_to_base(u, v, depth_m):
                                point_camera_mm = robot.pixel_to_camera(u, v, depth_m * 1000.0)
                                _, point_base_mm = robot.camera_to_base(
                                    point_camera_mm, tcp_pose=tcp_pose)
                                return np.asarray(point_base_mm, dtype=np.float64) / 1000.0
                            if (args.enable_angle_rotation or args.show_angle) and tool_category in (
                                    "screwdriver", "adjustable wrench", "rubber mallet",
                                    "tape dispenser"):
                                current_axis, angle_reason = principal_axis_base(
                                    res_hi["mask"], hi_depth * robot.camera.scale,
                                    hi_pixel_to_base)
                                if current_axis is not None:
                                    angle_history.append(current_axis)
                                    current_orientation = overhead_orientation(
                                        current_axis, tcp_pose[3:6], TOOL_ORIENTATION)
                                else:
                                    angle_history.clear()
                            if tool_category is not None:
                                current_support_plane, support_plane_reason = fit_horizontal_support_plane(
                                    hi_depth, robot.camera.scale, hi_pixel_to_base,
                                    exclude_mask=res_hi.get("mask"))
                            coord_name = "HI base"
                        except Exception:
                            x, y, z = hi_camera
                            coord_name = "HI camera"
                    else:
                        x, y, z = hi_camera
                        coord_name = "HI camera"
                    hi_coord = "%s  %s [%.3f, %.3f, %.3f]" % (
                        "PASS" if hi_gate["passed"] else "REJECT", coord_name, x, y, z)
            association = associate_targets(
                state["ho_base_aligned"], state["hi_base"], ho_gate, hi_gate,
                alignment_applied=camera_alignment is not None)
            state["association"] = association
            if current_axis is None:
                angle_history.clear()
            angle_stable = (len(angle_history) == angle_history.maxlen and
                            all(axial_difference_deg(angle_history[0], item)
                                <= ANGLE_STABLE_DEG for item in angle_history))
            approach_destination, approach_reason = safe_approach_candidate(association)
            if ENABLE_SAFE_APPROACH_TEST and approach_destination is not None:
                approach_ready_streak += 1
            else:
                approach_ready_streak = 0
            coarse_destination, coarse_reason = coarse_approach_candidate(
                state["ho_base_aligned"], ho_gate, camera_alignment is not None)
            if ENABLE_SAFE_APPROACH_TEST and coarse_destination is not None:
                coarse_ready_streak += 1
            else:
                coarse_ready_streak = 0
            handle_thickness = None
            if hi_handle is not None:
                handle_thickness, _ = estimate_handle_thickness(
                    hi_handle.radius_px, hi_handle.depth_m,
                    float(robot.cam_intrinsics[0, 0]))
            grasp_preview = build_grasp_preview(
                state["hi_base"], hi_gate, observation_active,
                support_plane=current_support_plane,
                calibration=screwdriver_calibration,
                handle_thickness_m=handle_thickness,
                profile=profile,
                orientation=(current_orientation if args.enable_angle_rotation
                             and angle_stable else None))
            if observation_active and grasp_preview.get("ready"):
                descent_ready_streak += 1
            else:
                descent_ready_streak = 0

            # 一键抓取：a 粗定位后自动推进 D 与 R，任一步超时/失败即停在安全位置。
            if auto_grasp_state == "wait_rotation":
                if time.time() > auto_grasp_deadline:
                    print("[一键抓取] 工具方向未稳定，停在观察点。")
                    auto_grasp_state = "idle"
                elif grasp_preview.get("ready") and angle_stable and current_orientation:
                    if rotate_to_planar_orientation(robot, current_orientation):
                        hi_filter.reset()
                        angle_history.clear()
                        angle_rotation_completed = True
                        auto_grasp_state = "wait_descent"
                        auto_grasp_deadline = time.time() + ANGLE_GRASP_TIMEOUT_S
                    else:
                        auto_grasp_state = "idle"
            elif auto_grasp_state == "wait_descent":
                if time.time() > auto_grasp_deadline:
                    print("[一键抓取] 超时：D435i 未稳定，停在观察点，不下降。")
                    auto_grasp_state = "idle"
                elif (not args.enable_angle_rotation or angle_rotation_completed) and (
                      descent_ready_streak >= SAFE_DESCENT_CONFIRM_FRAMES
                      and grasp_preview.get("ready")):
                    if args.enable_angle_rotation and (not current_orientation or
                            _orientation_error_deg(_read_valid_tcp_pose(robot)[3:6],
                                                   current_orientation) > ANGLE_STABLE_DEG):
                        print("[一键抓取] 旋转后方向不一致，禁止下降。")
                        auto_grasp_state = "idle"
                        continue
                    locked = attempt_descent(robot, grasp_preview, current_support_plane)
                    if locked is None:
                        print("[一键抓取] 下降未通过安全门，停在观察点。")
                        auto_grasp_state = "idle"
                    else:
                        locked_grasp_preview = dict(grasp_preview)
                        locked_screwdriver_handle = locked
                        safe_descent_completed = True
                        auto_grasp_state = "wait_grasp"
                        auto_grasp_deadline = time.time() + AUTO_GRASP_TIMEOUT_S
                        print("[一键抓取] 已到达无接触终点，等待夹持条件。")
            elif auto_grasp_state == "wait_grasp":
                if time.time() > auto_grasp_deadline:
                    print("[一键抓取] 超时：夹持条件未满足，停在下降终点。")
                    auto_grasp_state = "idle"
                else:
                    status, message = attempt_grasp(
                        robot, locked_screwdriver_handle, hi_gate, hi_handle,
                        current_support_plane, screwdriver_calibration, state["hi_base"],
                        grip_profile=profile)
                    if status == "done":
                        if placement is not None:
                            try:
                                place_at_fixed_point(robot, placement, WORKSPACE_LIMITS,
                                                     (profile["open_position"] if profile else GRIP_OPEN_POS),
                                                     GRIP_OPEN_SPEED,
                                                     GRIP_OPEN_FORCE)
                                print("[一键抓取] 已在固定位置放下工具。")
                            except Exception as exc:
                                print("[放置中止] %s；请检查机械臂和工具状态。" % exc)
                        else:
                            print("[一键抓取] 已夹起并停住。按 o 可松开。")
                        auto_grasp_state = "idle"
                    elif status == "failed":
                        print("[一键抓取] " + (message or "夹持失败"))
                        auto_grasp_state = "idle"
                    # status == "waiting"：条件未满足，继续等待（直到超时）

            if observation_active:
                association_text = grasp_preview_status_text(
                    grasp_preview, ENABLE_SAFE_DESCENT_TEST, descent_ready_streak,
                    safe_descent_completed)
                locked_pixel = None
                if locked_grasp_preview is not None:
                    display_tcp = tcp_pose
                    if display_tcp is None:
                        try:
                            display_tcp = _read_valid_tcp_pose(robot)
                        except Exception:
                            display_tcp = None
                    if display_tcp is not None:
                        locked_pixel = project_base_point_to_hi_pixel(
                            robot, locked_grasp_preview["target_surface_xyz_m"],
                            display_tcp, hi_disp.shape[:2])
                draw_grasp_preview(hi_disp, res_hi, grasp_preview,
                                   locked_pixel=locked_pixel)
            else:
                association_text = association_status_text(
                    association, ENABLE_SAFE_APPROACH_TEST,
                    approach_ready_streak, approach_reason)
            if current_axis is not None:
                cv2.putText(hi_disp, "BASE AXIS %.1f deg %s" %
                            (np.degrees(current_axis), "STABLE" if angle_stable else "TRACKING"),
                            (12, hi_disp.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
            draw_header(hi_disp, "D435I WRIST [%s:%d]" % (position_text, sample_count),
                        hi_status, hi_coord, gate_reason_text(hi_gate), association_text)
            draw_header(ho_disp, "D455 GLOBAL [%s:%d]" % (position_text, sample_count),
                        ho_status, ho_coord, gate_reason_text(ho_gate), association_text)
            cv2.imshow(win_ho, ho_disp)
            cv2.imshow(win_hi, hi_disp)

            both_ready = bool(ho_gate["passed"] and hi_gate["passed"])
            now = time.monotonic()
            if (active_position is not None and both_ready
                    and now - last_log_time >= MEASUREMENT_LOG_INTERVAL_S):
                append_validation_measurement(
                    log_path=validation_log,
                    session_id=session_id,
                    position_label=active_position,
                    ho_result=res_ho,
                    ho_base=state["ho_base"],
                    ho_base_aligned=state["ho_base_aligned"],
                    hi_result=res_hi,
                    hi_camera=hi_camera,
                    hi_base=state["hi_base"],
                    tcp_pose=tcp_pose,
                    association=association,
                )
                last_log_time = now
                position_counts[active_position] += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in POSITION_LABELS:
                active_position = POSITION_LABELS[key]
                position_counts[active_position] = 0
                last_log_time = 0.0
                ho_filter.reset()
                hi_filter.reset()
                print("[位置标记] %s：已清空旧稳定历史，等待两台相机重新 STABLE + PASS。"
                      % active_position)
            elif key == ord('o'):
                if gripper_enabled:
                    open_position = profile["open_position"] if profile else GRIP_OPEN_POS
                    robot.grip(open_position, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
                    print("[夹爪] 张开 pos=%d" % open_position)
                else:
                    print("[安全锁定] 夹爪未启用，无法控制。")
            elif key == ord('c'):
                if gripper_enabled:
                    close_position = profile["close_position"] if profile else GRIP_CLOSE_POS
                    robot.grip(close_position, GRIP_SPEED, GRIP_FORCE)
                    print("[夹爪] 闭合 pos=%d" % close_position)
                else:
                    print("[安全锁定] 夹爪未启用，无法控制。")
            elif key == ord('t'):
                if ENABLE_ROBOT_GRASP:
                    print("[姿态测试] 仅执行安全升高和末端摆正，不执行抓取")
                    normalize_tool_pose(robot)
                else:
                    print("[安全锁定] 纯视觉模式不发送机械臂运动命令。")
            elif key == ord('a'):
                if not ENABLE_SAFE_APPROACH_TEST:
                    print("[安全锁定] 请先将 ENABLE_SAFE_APPROACH_TEST 改为 True；默认不允许机械臂运动。")
                elif coarse_destination is None:
                    print("[安全锁定] D455 粗定位尚不可用：%s" % coarse_reason)
                elif coarse_ready_streak < SAFE_APPROACH_CONFIRM_FRAMES:
                    print("[安全锁定] D455 稳定帧不足：%d/%d。请保持目标静止。" %
                          (coarse_ready_streak, SAFE_APPROACH_CONFIRM_FRAMES))
                else:
                    print("[D455粗定位] 仅用D455对齐坐标移动到目标上方；腕部相机此时可能还看不到工具。")
                    if move_to_safe_observation(robot, coarse_destination):
                        angle_rotation_completed = False
                        d455_detection_frozen = True
                        observation_active = True
                        ho_filter.reset()
                        if ENABLE_ONE_KEY_GRASP and (tool_category == "screwdriver" or profile is not None):
                            auto_grasp_state = ("wait_rotation" if args.enable_angle_rotation
                                                else "wait_descent")
                            auto_grasp_deadline = time.time() + (
                                ANGLE_GRASP_TIMEOUT_S if args.enable_angle_rotation
                                else AUTO_GRASP_TIMEOUT_S)
                            print("[一键抓取] 已进入自动模式：等 D435i 稳定后自动下降并夹持。")
                        else:
                            print("[D455粗定位完成] 已停在目标上方；仅螺丝刀有自动抓取路径。")
            elif key == ord('p'):
                if not ENABLE_SAFE_APPROACH_TEST:
                    print("[安全锁定] 请先将 ENABLE_SAFE_APPROACH_TEST 改为 True；默认不允许机械臂运动。")
                elif approach_destination is None:
                    print("[安全锁定] 双相机尚未通过安全观察点门槛：%s" % approach_reason)
                elif approach_ready_streak < SAFE_APPROACH_CONFIRM_FRAMES:
                    print("[安全锁定] 双相机一致帧不足：%d/%d。继续保持目标静止。" %
                          (approach_ready_streak, SAFE_APPROACH_CONFIRM_FRAMES))
                else:
                    if move_to_safe_observation(robot, approach_destination):
                        angle_rotation_completed = False
                        d455_detection_frozen = True
                        observation_active = True
                        ho_filter.reset()
                        print("[观察点] 已暂停D455目标检测，避免移动中的夹爪被识别为工具。")
                        print("[抓取预览] D435i 将持续显示预抓取点和虚拟下降终点；不会发送运动或夹爪命令。")
            elif key == ord('d'):
                if not ENABLE_SAFE_DESCENT_TEST:
                    print("[安全锁定] 将 ENABLE_SAFE_DESCENT_TEST 改为 True 后才允许无接触下降测试。")
                elif safe_descent_completed:
                    print("[安全锁定] 本次运行已完成一次无接触下降；请重启程序后再测试。")
                elif not observation_active:
                    print("[安全锁定] 请先完成 a（D455粗定位）或 P（双相机观察点）。")
                elif args.enable_angle_rotation and not angle_rotation_completed:
                    print("[安全锁定] 请先按 y 在安全高度旋转并等待 D435i 重新稳定。")
                elif args.enable_angle_rotation and (not angle_stable or not current_orientation or
                        _orientation_error_deg(_read_valid_tcp_pose(robot)[3:6],
                                               current_orientation) > ANGLE_STABLE_DEG):
                    print("[安全锁定] 旋转后方向未重新稳定或与当前姿态不一致。")
                elif not grasp_preview.get("ready"):
                    print("[安全锁定] D435i 预览无效：%s" % grasp_preview.get("reason", "unknown"))
                elif descent_ready_streak < SAFE_DESCENT_CONFIRM_FRAMES:
                    print("[安全锁定] D435i 稳定帧不足：%d/%d。" %
                          (descent_ready_streak, SAFE_DESCENT_CONFIRM_FRAMES))
                else:
                    locked = attempt_descent(robot, grasp_preview, current_support_plane)
                    if locked is not None:
                        locked_grasp_preview = dict(grasp_preview)
                        locked_screwdriver_handle = locked
                        safe_descent_completed = True
                        print("[无接触下降] 已锁定D键触发时的目标中心；后续画面不再跟随分割中心漂移。")
            elif key == ord('r'):
                if not ENABLE_SCREWDRIVER_GRASP:
                    print("[安全锁定] 此工具尚未启用经过审核的抓取配置。")
                elif tool_category == "screwdriver" and screwdriver_calibration is None:
                    print("[安全锁定] 未找到有效 grasp_surface_calibration.json。")
                elif not safe_descent_completed or locked_screwdriver_handle is None:
                    print("[安全锁定] 请先完成 P 和 D 无接触下降测试。")
                elif "grasp_tcp_z_m" not in locked_screwdriver_handle:
                    print("[安全锁定] D 未记录自适应夹持高度；请重新运行 P → D。")
                elif not (hi_gate["passed"] and (hi_handle is not None or profile is not None)
                          and current_support_plane is not None):
                    print("[安全锁定] 等待稳定手柄区域和当前支撑面。")
                else:
                    status, message = attempt_grasp(
                        robot, locked_screwdriver_handle, hi_gate, hi_handle,
                        current_support_plane, screwdriver_calibration, state["hi_base"],
                        grip_profile=profile)
                    if message:
                        print("[安全锁定] " + message)
                    if status == "done" and placement is not None:
                        try:
                            place_at_fixed_point(robot, placement, WORKSPACE_LIMITS,
                                                 (profile["open_position"] if profile else GRIP_OPEN_POS),
                                                 GRIP_OPEN_SPEED,
                                                 GRIP_OPEN_FORCE)
                            print("[固定放置] 完成。")
                        except Exception as exc:
                            print("[放置中止] %s；请检查机械臂和工具状态。" % exc)
            elif key == ord('y'):
                if not args.enable_angle_rotation or not observation_active:
                    print("[安全锁定] 使用 --stage rotate 或后续阶段并先到达观察点。")
                elif not (angle_stable and current_orientation and grasp_preview.get("ready")):
                    print("[安全锁定] D435i 方向或抓取预览尚未稳定。")
                elif rotate_to_planar_orientation(robot, current_orientation):
                    hi_filter.reset()
                    angle_history.clear()
                    angle_rotation_completed = True
                    print("[方向对齐] 已旋转；等待 D435i 重新定位后再按 D。")
    finally:
        if getattr(robot, "camera", None) is not None:
            robot.camera.stop()
        robot_state.close()
        ho_cam.stop()
        cv2.destroyAllWindows()
        print("退出。")


def draw_header(image, camera_name, status, coordinate=None, gate_reason=None,
                association_text=None):
    """Draw non-overlapping camera, tracking status and coordinate lines."""
    colors = {
        "SEARCHING": (0, 0, 255),
        "TRACKING": (0, 200, 255),
        "STABLE": (0, 255, 0),
    }
    cv2.rectangle(image, (0, 0), (image.shape[1], 114), (20, 20, 20), -1)
    cv2.putText(image, "%s  %s" % (camera_name, status), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, colors.get(status, (255, 255, 255)),
                2, cv2.LINE_AA)
    detail = coordinate or "coordinate unavailable until STABLE"
    cv2.putText(image, detail, (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (230, 230, 230), 1, cv2.LINE_AA)
    if gate_reason:
        cv2.putText(image, gate_reason, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (180, 180, 180), 1, cv2.LINE_AA)
    if association_text:
        color = ((0, 255, 0) if association_text.startswith(("SAME TARGET", "APPROACH READY"))
                 else (0, 200, 255))
        cv2.putText(image, association_text, (10, 103), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, color, 1, cv2.LINE_AA)


def associate_targets(ho_base, hi_base, ho_gate, hi_gate, alignment_applied=False):
    """Compare independent base-frame estimates; never authorize robot motion."""
    result = {
        "available": False,
        "matched": False,
        "distance_m": None,
        "target_base_xyz_m": None,
        "approach_base_xyz_m": None,
        "robot_motion_authorized": False,
        "alignment_applied": bool(alignment_applied),
    }
    if not (ho_gate["passed"] and hi_gate["passed"]):
        result["reason"] = "waiting for both validation gates"
        return result
    if ho_base is None or hi_base is None:
        result["reason"] = "read-only TCP unavailable"
        return result

    distance = float(np.linalg.norm(np.asarray(ho_base) - np.asarray(hi_base)))
    result["available"] = True
    result["distance_m"] = distance
    result["matched"] = distance <= TARGET_ASSOCIATION_MAX_DISTANCE_M
    if not result["matched"]:
        result["reason"] = "base coordinates disagree"
        return result

    target = np.asarray(hi_base, dtype=np.float64)
    approach = target.copy()
    approach[2] = min(target[2] + APPROACH_HEIGHT_M, WORKSPACE_LIMITS[2][1])
    result["target_base_xyz_m"] = target.tolist()
    result["approach_base_xyz_m"] = approach.tolist()
    result["reason"] = "same prompt and nearby base coordinates"
    return result


def association_status_text(association, safe_approach_mode=False,
                            ready_streak=0, approach_reason=None):
    if not association["available"]:
        return "ASSOCIATION WAITING: " + association.get("reason", "unavailable")
    distance_mm = association["distance_m"] * 1000.0
    source = "aligned" if association.get("alignment_applied") else "raw"
    if safe_approach_mode:
        if approach_reason is None and ready_streak >= SAFE_APPROACH_CONFIRM_FRAMES:
            return "APPROACH READY %d/%d  delta=%.1fmm  press P" % (
                ready_streak, SAFE_APPROACH_CONFIRM_FRAMES, distance_mm)
        if approach_reason is None:
            return "APPROACH CHECK %d/%d  delta=%.1fmm" % (
                ready_streak, SAFE_APPROACH_CONFIRM_FRAMES, distance_mm)
        return "APPROACH LOCKED: " + approach_reason
    if association["matched"]:
        return "SAME TARGET  %s delta=%.1fmm  preview only" % (source, distance_mm)
    return "TARGET MISMATCH  %s delta=%.1fmm" % (source, distance_mm)


def point_in_workspace(point):
    """Return whether an XYZ base-frame point is strictly inside the configured workspace."""
    return all(float(low) <= float(value) <= float(high)
               for value, (low, high) in zip(point, WORKSPACE_LIMITS))


def safe_approach_candidate(association):
    """Build a no-descent observation pose only after strict dual-camera agreement.

    The regular association threshold remains intentionally looser for visual
    diagnostics. Motion uses this separate 15 mm threshold and requires the
    saved D455/D435 alignment, so a raw D455 coordinate can never authorize P.
    """
    if not association or not association.get("available"):
        return None, "waiting for both cameras"
    if not association.get("alignment_applied"):
        return None, "camera alignment file not loaded"
    if not association.get("matched"):
        return None, "two cameras do not identify one target"
    distance = association.get("distance_m")
    if distance is None or float(distance) > SAFE_APPROACH_ASSOCIATION_MAX_DISTANCE_M:
        return None, "camera delta exceeds %.0fmm" % (
            SAFE_APPROACH_ASSOCIATION_MAX_DISTANCE_M * 1000.0)
    target = association.get("target_base_xyz_m")
    if target is None or not point_in_workspace(target):
        return None, "target outside workspace"

    destination = np.asarray(target, dtype=np.float64).copy()
    destination[2] = max(float(target[2]) + APPROACH_HEIGHT_M, SAFE_TRAVEL_Z_M)
    if not point_in_workspace(destination):
        return None, "safe observation point outside workspace"
    return destination.tolist(), None


def coarse_approach_candidate(ho_base_aligned, ho_gate, alignment_applied):
    """Build a D455-only coarse observation pose when the wrist camera cannot see the tool.

    Used at startup, before the arm has moved, so the wrist D435i has no target
    yet.  It only moves to a high observation point; D still requires the D435i
    to confirm the tool before any descent.  The saved D455/D435 alignment is
    mandatory because the raw D455 coordinate carries a systematic offset.
    """
    if not ho_gate or not ho_gate.get("passed") or ho_base_aligned is None:
        return None, "waiting for stable D455 target"
    if not alignment_applied:
        return None, "camera alignment file not loaded"
    target = np.asarray(ho_base_aligned, dtype=np.float64)
    if not point_in_workspace(target):
        return None, "D455 target outside workspace"

    destination = target.copy()
    destination[2] = max(float(target[2]) + APPROACH_HEIGHT_M, SAFE_TRAVEL_Z_M)
    if not point_in_workspace(destination):
        return None, "coarse observation point outside workspace"
    return destination.tolist(), None


def build_grasp_preview(hi_base, hi_gate, observation_active,
                        support_plane=None, calibration=None,
                        handle_thickness_m=None, orientation=None, profile=None):
    """Plan the descent to the handle mid using the adaptive grasp model.

    The handle thickness is estimated from its projected width every attempt, and
    the tool-independent ``gripper_offset_m`` converts the handle mid height into
    a TCP height.  A different screwdriver therefore needs no new calibration.
    """
    preview = {"active": bool(observation_active), "ready": False,
               "orientation": orientation or TOOL_ORIENTATION}
    if not observation_active:
        return preview
    if not hi_gate["passed"] or hi_base is None:
        preview["reason"] = "waiting for stable D435i target"
        return preview
    target = np.asarray(hi_base, dtype=np.float64)
    if not point_in_workspace(target):
        preview["reason"] = "D435i target outside workspace"
        return preview

    plan = None
    grasp_tcp_z = None
    if profile is not None:
        if support_plane is None:
            preview["reason"] = "support plane unavailable for tool profile"
            return preview
        grasp_tcp_z = float(profile["grasp_tcp_z_m"])
        if not 0.005 <= grasp_tcp_z - support_plane.z_m <= 0.25:
            preview["reason"] = "profile grip height inconsistent with current support plane"
            return preview
        plan = {"grasp_tcp_z_m": grasp_tcp_z,
                "support_plane_z_m": support_plane.z_m,
                "profile_grasp": True}
    elif (calibration is not None and support_plane is not None
            and handle_thickness_m is not None):
        grasp_tcp_z, plan = plan_grasp_tcp(
            float(handle_thickness_m), float(support_plane.z_m),
            float(calibration["gripper_offset_m"]))

    if grasp_tcp_z is None:
        # Fall back to a visual-only preview (no adaptive grasp geometry).
        endpoint = target.copy()
        pregrasp = target.copy()
        pregrasp[2] = min(target[2] + GRASP_PREVIEW_HEIGHT_M, WORKSPACE_LIMITS[2][1])
        if not point_in_workspace(endpoint) or not point_in_workspace(pregrasp):
            preview["reason"] = "virtual point outside workspace"
            return preview
        if pregrasp[2] <= endpoint[2]:
            preview["reason"] = "virtual descent has no clearance"
            return preview
        preview.update({
            "ready": True,
            "adaptive": False,
            "reason": plan if isinstance(plan, str) else "grasp plan unavailable",
            "target_surface_xyz_m": target.tolist(),
            "pregrasp_xyz_m": pregrasp.tolist(),
            "endpoint_xyz_m": endpoint.tolist(),
            "descent_m": float(pregrasp[2] - endpoint[2]),
        })
        return preview

    endpoint = target.copy()
    endpoint[2] = grasp_tcp_z
    pregrasp = target.copy()
    pregrasp[2] = min(grasp_tcp_z + GRASP_PREVIEW_HEIGHT_M, WORKSPACE_LIMITS[2][1])
    if not point_in_workspace(endpoint) or not point_in_workspace(pregrasp):
        preview["reason"] = "grasp point outside workspace"
        return preview
    if pregrasp[2] <= endpoint[2]:
        preview["reason"] = "grasp descent has no clearance"
        return preview
    preview.update({
        "ready": True,
        "adaptive": True,
        "plan": plan,
        "target_surface_xyz_m": target.tolist(),
        "pregrasp_xyz_m": pregrasp.tolist(),
        "endpoint_xyz_m": endpoint.tolist(),
        "descent_m": float(pregrasp[2] - endpoint[2]),
    })
    return preview


def grasp_preview_status_text(preview, safe_descent_enabled=False, ready_streak=0,
                              safe_descent_completed=False):
    if not preview.get("ready"):
        return "GRASP PREVIEW: " + preview.get("reason", "waiting")
    endpoint = preview["endpoint_xyz_m"]
    if safe_descent_enabled:
        if safe_descent_completed:
            return "SAFE DESCENT COMPLETE: locked center, no further motion"
        if ready_streak >= SAFE_DESCENT_CONFIRM_FRAMES:
            return "SAFE DESCENT READY %d/%d: press D, stop %.0fmm above tool" % (
                ready_streak, SAFE_DESCENT_CONFIRM_FRAMES,
                SAFE_DESCENT_CLEARANCE_M * 1000.0)
        return "SAFE DESCENT CHECK %d/%d: D435i must remain stable" % (
            ready_streak, SAFE_DESCENT_CONFIRM_FRAMES)
    return "GRASP PREVIEW ONLY: end [%.3f, %.3f, %.3f], descend %.0fmm" % (
        endpoint[0], endpoint[1], endpoint[2], preview["descent_m"] * 1000.0)


def project_base_point_to_hi_pixel(robot, base_xyz_m, tcp_pose, image_shape):
    """Project a locked base-frame point into the current D435i image.

    This is the inverse of camera_to_base and uses the current TCP, so the
    marker stays attached to the same physical base-frame point as the wrist
    camera moves.
    """
    try:
        target_base_mm = np.asarray(base_xyz_m, dtype=np.float64).reshape(3) * 1000.0
        target_h = np.append(target_base_mm, 1.0)
        t_end_to_base = robot.pose_vector_to_matrix(tcp_pose)
        t_cam_to_base = t_end_to_base @ robot.T_cam2end
        point_camera = np.linalg.inv(t_cam_to_base) @ target_h
        if point_camera[2] <= 1.0:
            return None
        k = np.asarray(robot.cam_intrinsics, dtype=np.float64)
        u = int(round(k[0, 0] * point_camera[0] / point_camera[2] + k[0, 2]))
        v = int(round(k[1, 1] * point_camera[1] / point_camera[2] + k[1, 2]))
        height, width = image_shape
        return (u, v) if 0 <= u < width and 0 <= v < height else None
    except Exception:
        return None


def draw_grasp_preview(image, result, preview, locked_pixel=None):
    """Overlay the virtual center; use a reprojected locked point after D."""
    if locked_pixel is not None:
        u, v = locked_pixel
        label = "locked grasp center"
    elif preview.get("ready") and result is not None:
        u, v = [int(value) for value in result["center"]]
        label = "preview grasp center"
    else:
        return
    color = (255, 255, 0)
    cv2.circle(image, (u, v), 14, color, 2)
    cv2.line(image, (u - 20, v), (u + 20, v), color, 1)
    cv2.line(image, (u, v - 20), (u, v + 20), color, 1)
    cv2.putText(image, label, (u + 18, v + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def gate_detection(result, tracking_status, camera_name, base_xyz=None):
    """Check whether a stable visual result is trustworthy enough to record."""
    reasons = []
    if tracking_status != "STABLE" or result is None:
        reasons.append("not stable")
        return {"passed": False, "reasons": reasons}

    min_score = D455_MIN_SCORE if camera_name == "D455" else D435I_MIN_SCORE
    if float(result.get("score", 0.0)) < min_score:
        reasons.append("score < %.2f" % min_score)
    x1, y1, x2, y2 = result.get("box", (0, 0, 0, 0))
    if min(x2 - x1, y2 - y1) < MIN_BOX_SIDE_PX:
        reasons.append("box too small")
    depth_points = int(result.get("valid_depth_points", 0))
    if depth_points < MIN_VALID_DEPTH_POINTS:
        reasons.append("too few depth points (%d<%d)" % (depth_points, MIN_VALID_DEPTH_POINTS))
    if result.get("z_mm") is None:
        reasons.append("invalid depth")

    if base_xyz is not None:
        for axis, value, limits in zip("XYZ", base_xyz, WORKSPACE_LIMITS):
            if not (float(limits[0]) <= float(value) <= float(limits[1])):
                reasons.append("%s outside validation workspace" % axis)
    return {"passed": not reasons, "reasons": reasons}


def gate_reason_text(gate):
    if gate["passed"]:
        return "validation gate: PASS (recording measurement)"
    return "validation gate: " + ", ".join(gate["reasons"])


def _serializable_result(result):
    if result is None:
        return None
    return {
        "prompt": result.get("prompt"),
        "score": float(result.get("score", 0.0)),
        "sam_score": float(result.get("sam_score", 0.0)),
        "center_px": [int(v) for v in result.get("center", (0, 0))],
        "box_px": [int(v) for v in result.get("box", (0, 0, 0, 0))],
        "depth_m": float(result["z_mm"]) / 1000.0,
        "angle_deg": float(result.get("angle_deg", 0.0)),
        "valid_depth_points": int(result.get("valid_depth_points", 0)),
        "center_spread_px": float(result.get("center_spread_px", 0.0)),
        "depth_spread_mm": float(result.get("depth_spread_mm", 0.0)),
    }


def append_validation_measurement(log_path, session_id, position_label,
                                  ho_result, ho_base, ho_base_aligned,
                                  hi_result, hi_camera,
                                  hi_base=None, tcp_pose=None, association=None):
    """Append one compact record; JSONL survives interruption and is easy to compare."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "session_id": session_id,
        "position_label": position_label,
        "prompt": TEXT_PROMPT,
        "vision_gate_passed": True,
        "cross_camera_same_target_verified": bool(
            association and association.get("matched")),
        "robot_motion_authorized": False,
        "association": association,
        "d455": {
            "serial": HO_SERIAL,
            "result": _serializable_result(ho_result),
            "base_xyz_m": [float(v) for v in ho_base] if ho_base is not None else None,
            "base_xyz_aligned_m": ([float(v) for v in ho_base_aligned]
                                   if ho_base_aligned is not None else None),
        },
        "d435i": {
            "serial": HI_SERIAL,
            "result": _serializable_result(hi_result),
            "camera_xyz_m": [float(v) for v in hi_camera] if hi_camera is not None else None,
            "base_xyz_m": [float(v) for v in hi_base] if hi_base is not None else None,
            "tcp_pose_m_rad": [float(v) for v in tcp_pose] if tcp_pose is not None else None,
        },
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_calib(robot, ho_cam):
    """比对实时内参与标定内参，超阈值则抛错拒绝执行。"""
    ok = True

    # ---- 手外 D455 ----
    live = ho_cam.intrinsics
    exp = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    dho = max_intrinsics_delta(live, exp)
    print("手外D455 实时 fx=%.3f fy=%.3f cx=%.3f cy=%.3f | 期望=%.3f/%.3f/%.3f/%.3f | Δmax=%.3f px"
          % (live[0, 0], live[1, 1], live[0, 2], live[1, 2],
             exp[0, 0], exp[1, 1], exp[0, 2], exp[1, 2], dho))
    if not np.all(np.isfinite(live)):
        print("[自检失败] 手外D455 实时内参包含非有限值。")
        ok = False
    elif dho > D455_CALIB_TOL:
        print("[自检失败] 手外D455 内参与标定值不符(%.3f>%.1f)！"
              % (dho, D455_CALIB_TOL))
        ok = False

    # ---- 手内 D435I ----
    calib = load_camera_ini(CAM_INI)
    if calib is None:
        print("[警告] 未找到/无法解析 %s，手内内参无法校验" % CAM_INI)
    else:
        K_ini, _ = calib
        live_hi = robot.camera.intrinsics
        connected_serial = getattr(robot.camera, "connected_serial", None)
        if connected_serial != HI_SERIAL:
            print("[自检失败] 手内相机序列号不符(实际=%s，期望=%s)。"
                  % (connected_serial or "未知", HI_SERIAL))
            ok = False
        if (robot.camera.im_width, robot.camera.im_height) != (640, 480):
            print("[自检失败] 手内D435i 图像流不是 640x480 (当前=%sx%s)。"
                  % (robot.camera.im_width, robot.camera.im_height))
            ok = False
        if not np.all(np.isfinite(live_hi)):
            print("[自检失败] 手内D435i 实时内参包含非有限值。")
            ok = False
            dhi = float("inf")
        else:
            dhi = max_intrinsics_delta(live_hi, K_ini)
        print("手内D435I 实时 fx=%.3f fy=%.3f cx=%.3f cy=%.3f | 期望=%.3f/%.3f/%.3f/%.3f | Δmax=%.3f px"
              % (live_hi[0, 0], live_hi[1, 1], live_hi[0, 2], live_hi[1, 2],
                 K_ini[0, 0], K_ini[1, 1], K_ini[0, 2], K_ini[1, 2], dhi))
        if dhi > D435I_CALIB_TOL:
            print("[自检失败] 手内D435I 内参与标定值不符(%.3f>%.1f)！"
                  % (dhi, D435I_CALIB_TOL))
            ok = False
        elif dhi > D435I_VERIFIED_WARNING_TOL:
            print("[自检警告] 手内D435i 内参差异 %.3fpx：已按当前验证的 640x480 配置放行 "
                  "(上限 %.1fpx)。请勿更换相机、分辨率或手眼标定。"
                  % (dhi, D435I_CALIB_TOL))

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


def verify_tool_orientation(robot, stage, expected=None):
    """检查末端是否到达标准朝下姿态；读取或误差异常时返回False。"""
    try:
        actual = _read_valid_tcp_pose(robot)
        error_deg = _orientation_error_deg(actual[3:6],
                                           TOOL_ORIENTATION if expected is None else expected)
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


def rotate_to_planar_orientation(robot, orientation):
    """Rotate only at the high observation pose; the wrist camera must reobserve."""
    try:
        current = _read_valid_tcp_pose(robot)
        if current[2] < SAFE_TRAVEL_Z_M - 0.005:
            print("[安全中止] 低于安全观察高度，禁止旋转。")
            return False
        target = current.copy()
        target[3:6] = orientation
        robot.moveL(target.tolist(), speed=SAFE_APPROACH_SPEED,
                    acceleration=SAFE_APPROACH_ACCELERATION)
        return verify_tool_orientation(robot, "平面方向对齐", orientation)
    except Exception as exc:
        print("[安全中止] 方向对齐失败：%s" % exc)
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


def move_to_safe_observation(robot, destination_xyz):
    """Move only to a high observation point; this function never descends or uses the gripper."""
    destination = np.asarray(destination_xyz, dtype=np.float64).reshape(3)
    if not point_in_workspace(destination):
        print("[安全中止] 安全观察点超出工作空间：%s" %
              ["%.4f" % value for value in destination])
        return False
    if destination[2] < SAFE_TRAVEL_Z_M - 1e-6:
        print("[安全中止] 观察点低于安全通行高度，拒绝执行。")
        return False

    try:
        current = _read_valid_tcp_pose(robot)
        print("[观察点] 当前TCP=%s" % ["%.4f" % value for value in current])

        # First raise vertically, keeping the current XY and orientation.  This
        # prevents a horizontal sweep near the table or the target.
        lift_target = current.copy()
        lift_target[2] = max(float(current[2]), SAFE_TRAVEL_Z_M)
        if lift_target[2] > current[2] + 0.001:
            print("[观察点] 垂直抬升至 z=%.3fm" % lift_target[2])
            robot.moveL(lift_target.tolist(), speed=SAFE_APPROACH_SPEED,
                        acceleration=SAFE_APPROACH_ACCELERATION)
        else:
            print("[观察点] 当前TCP已高于安全通行高度，不执行下降。")

        # Rotate only at the safe height, then move horizontally at or above it.
        after_lift = _read_valid_tcp_pose(robot)
        straighten_target = after_lift.copy()
        straighten_target[2] = max(float(after_lift[2]), SAFE_TRAVEL_Z_M)
        straighten_target[3:6] = TOOL_ORIENTATION
        print("[观察点] 安全高度摆正末端")
        robot.moveL(straighten_target.tolist(), speed=SAFE_APPROACH_SPEED,
                    acceleration=SAFE_APPROACH_ACCELERATION)
        if not verify_tool_orientation(robot, "观察点摆正后"):
            return False

        observation_pose = destination.tolist() + TOOL_ORIENTATION
        print("[观察点] 水平移动到目标上方=%s" %
              ["%.4f" % value for value in observation_pose])
        robot.moveL(observation_pose, speed=SAFE_APPROACH_SPEED,
                    acceleration=SAFE_APPROACH_ACCELERATION)
        if not verify_tool_orientation(robot, "观察点到达后"):
            return False
        print("[观察点完成] 已停在目标上方；未下降、未发送夹爪命令、未执行抓取。")
        return True
    except Exception as exc:
        print("[安全中止] 安全观察点移动失败：%s" % exc)
        try:
            if robot.rtde_c is not None and hasattr(robot.rtde_c, "stopL"):
                robot.rtde_c.stopL(1.0)
        except Exception as stop_exc:
            print("[提示] 无法发送停止命令，请用示教器检查：%s" % stop_exc)
        return False


def move_to_safe_descent_test(robot, preview):
    """Perform one slow no-contact descent; never touch the tool or use the gripper."""
    if not preview.get("ready"):
        print("[安全中止] 无有效D435i预览，拒绝无接触下降。")
        return False
    target = np.asarray(preview["target_surface_xyz_m"], dtype=np.float64)
    grasp_point = np.asarray(preview["endpoint_xyz_m"], dtype=np.float64)
    pregrasp = np.asarray(preview["pregrasp_xyz_m"], dtype=np.float64)
    stop_point = grasp_point.copy()
    stop_point[2] += SAFE_DESCENT_CLEARANCE_M
    orientation = preview.get("orientation", TOOL_ORIENTATION)
    if preview.get("plan") and "handle_thickness_m" in preview["plan"]:
        plan = preview["plan"]
        print("[抓取几何] 手柄厚度=%.1fmm 手柄中段z=%.4f 夹爪偏移=%.1fmm 夹持TCP z=%.4f" % (
            plan["handle_thickness_m"] * 1000.0, plan["handle_mid_z_m"],
            plan["gripper_offset_m"] * 1000.0, plan["grasp_tcp_z_m"]))
    if not (point_in_workspace(pregrasp) and point_in_workspace(stop_point)):
        print("[安全中止] 无接触下降路径超出工作空间。")
        return False
    if stop_point[2] >= pregrasp[2]:
        print("[安全中止] 无接触终点不低于预抓取点，拒绝执行。")
        return False

    try:
        current = _read_valid_tcp_pose(robot)
        if current[2] < SAFE_TRAVEL_Z_M - 0.005:
            print("[安全中止] 当前TCP不在安全观察高度，拒绝下降测试。")
            return False
        if not verify_tool_orientation(robot, "无接触下降前", orientation):
            return False

        # Horizontal correction happens only at the already verified high point.
        high_align = current.copy()
        high_align[0:2] = target[0:2]
        high_align[2] = max(float(current[2]), SAFE_TRAVEL_Z_M)
        high_align[3:6] = orientation
        print("[无接触下降] 安全高度对准 XY=%s" %
              ["%.4f" % value for value in high_align[0:3]])
        robot.moveL(high_align.tolist(), speed=SAFE_DESCENT_SPEED,
                    acceleration=SAFE_DESCENT_ACCELERATION)

        pregrasp_pose = pregrasp.tolist() + list(orientation)
        print("[无接触下降] 下降到预抓取高度=%s" %
              ["%.4f" % value for value in pregrasp_pose])
        robot.moveL(pregrasp_pose, speed=SAFE_DESCENT_SPEED,
                    acceleration=SAFE_DESCENT_ACCELERATION)
        if not verify_tool_orientation(robot, "预抓取高度", orientation):
            return False

        stop_pose = stop_point.tolist() + list(orientation)
        print("[无接触下降] 低速停在夹持点上方 %.0fmm=%s" % (
            SAFE_DESCENT_CLEARANCE_M * 1000.0,
            ["%.4f" % value for value in stop_pose]))
        robot.moveL(stop_pose, speed=SAFE_DESCENT_SPEED,
                    acceleration=SAFE_DESCENT_ACCELERATION)
        if not verify_tool_orientation(robot, "无接触终点", orientation):
            return False
        print("[无接触下降完成] 已停在夹持点上方 %.0fmm；未接触工具、未控制夹爪。" %
              (SAFE_DESCENT_CLEARANCE_M * 1000.0))
        return True
    except Exception as exc:
        print("[安全中止] 无接触下降失败：%s" % exc)
        try:
            if robot.rtde_c is not None and hasattr(robot.rtde_c, "stopL"):
                robot.rtde_c.stopL(1.0)
        except Exception as stop_exc:
            print("[提示] 无法发送停止命令，请用示教器检查：%s" % stop_exc)
        return False


def execute_screwdriver_grasp(robot, target_xyz_m, grasp_tcp_z, support_plane_z_m,
                              calibration, orientation=None, grip_profile=None):
    """Perform the guarded final stage after a verified no-contact descent.

    ``grasp_tcp_z`` is planned from the live handle thickness and the one-time
    tool-independent gripper offset, so a different screwdriver needs no new
    calibration.  The D no-contact endpoint must be SAFE_DESCENT_CLEARANCE_M
    above this height; R descends the remaining distance and closes the gripper.
    """
    target = np.asarray(target_xyz_m, dtype=np.float64).reshape(3)
    orientation = TOOL_ORIENTATION if orientation is None else orientation
    open_position = (GRIP_OPEN_POS if grip_profile is None
                     else grip_profile["open_position"])
    close_position = (GRIP_CLOSE_POS if grip_profile is None
                      else grip_profile["close_position"])
    grip_force = (SCREWDRIVER_GRASP_FORCE if grip_profile is None
                  else grip_profile["grip_force"])
    torque_min = (GRIP_TORQUE_MIN if grip_profile is None
                  else grip_profile["torque_min"])
    final = target.copy()
    final[2] = float(grasp_tcp_z)
    plane_clearance = final[2] - float(support_plane_z_m)
    if plane_clearance < 0.003:
        print("[安全中止] 夹持TCP离支撑面仅 %.1fmm，过低，拒绝执行。" % (plane_clearance * 1000.0))
        return False
    retreat = final.copy()
    retreat[2] += SAFE_DESCENT_CLEARANCE_M
    if not (point_in_workspace(final) and point_in_workspace(retreat)):
        print("[安全中止] 螺丝刀夹持路径超出工作空间。")
        return False
    try:
        current = _read_valid_tcp_pose(robot)
        start_error_m = float(np.linalg.norm(current[:3] - retreat))
        if start_error_m > SCREWDRIVER_R_START_TOL_M:
            print("[安全中止] 当前TCP未停在D无接触终点（偏差 %.1fmm），拒绝执行R。" %
                  (start_error_m * 1000.0))
            return False
        if not verify_tool_orientation(robot, "螺丝刀夹持前", orientation):
            return False
        high_align = current.copy()
        high_align[:2] = target[:2]
        high_align[2] = max(float(current[2]), float(retreat[2]))
        high_align[3:6] = orientation
        if robot.grip(open_position, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE) == -1:
            raise RuntimeError("gripper did not open")
        print("[螺丝刀抓取] 在安全高度对准手柄中段。")
        robot.moveL(high_align.tolist(), speed=SCREWDRIVER_GRASP_SPEED,
                    acceleration=SCREWDRIVER_GRASP_SPEED)
        print("[螺丝刀抓取] 低速下降到夹持高度=%s" %
              ["%.4f" % value for value in final])
        robot.moveL(final.tolist() + list(orientation), speed=SCREWDRIVER_GRASP_SPEED,
                    acceleration=SCREWDRIVER_GRASP_SPEED)
        if not verify_tool_orientation(robot, "闭爪前", orientation):
            return False
        if robot.grip(close_position, GRIP_SPEED, grip_force) == -1:
            raise RuntimeError("gripper did not close")
        torque_reached = robot.read_torque_reached()
        torque_current = robot.read_torque_current()
        contact_confirmed = torque_reached == 1 or torque_current >= torque_min
        if not contact_confirmed:
            print("[夹持失败] reached=%s torque=%s；张开并退回安全高度，不抬升工具。" %
                  (torque_reached, torque_current))
            robot.grip(open_position, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
            robot.moveL(retreat.tolist() + list(orientation), speed=SCREWDRIVER_GRASP_SPEED,
                        acceleration=SCREWDRIVER_GRASP_SPEED)
            return False
        lift = final.copy()
        lift[2] += SCREWDRIVER_TEST_LIFT_M
        if not point_in_workspace(lift):
            print("[安全中止] %.0fmm 测试抬升点超出工作空间；不抬升。"
                  % (SCREWDRIVER_TEST_LIFT_M * 1000.0))
            return False
        robot.moveL(lift.tolist() + list(orientation), speed=SCREWDRIVER_GRASP_SPEED,
                    acceleration=SCREWDRIVER_GRASP_SPEED)
        print("[螺丝刀抓取完成] reached=%s torque=%s；已低力夹持并抬升 %.0fmm，停在抬升位置。"
              % (torque_reached, torque_current, SCREWDRIVER_TEST_LIFT_M * 1000.0))
        return True
    except Exception as exc:
        print("[安全中止] 螺丝刀抓取异常：%s" % exc)
        try:
            if robot.rtde_c is not None and hasattr(robot.rtde_c, "stopL"):
                robot.rtde_c.stopL(1.0)
        except Exception:
            pass
        return False


def attempt_descent(robot, grasp_preview, current_support_plane):
    """Run one guarded no-contact descent and return the locked handle, or None.

    Shared by the manual D key and the one-key auto grasp so both use the same
    safety checks.
    """
    if not move_to_safe_descent_test(robot, grasp_preview):
        return None
    locked = {
        "target_base_xyz_m": list(grasp_preview["target_surface_xyz_m"]),
        "support_plane_z_m": (current_support_plane.z_m
                              if current_support_plane is not None else None),
        "orientation": list(grasp_preview.get("orientation", TOOL_ORIENTATION)),
    }
    if grasp_preview.get("adaptive"):
        locked["grasp_tcp_z_m"] = float(grasp_preview["endpoint_xyz_m"][2])
    return locked


def attempt_grasp(robot, locked_handle, hi_gate, hi_handle, current_support_plane,
                  calibration, current_target, grip_profile=None):
    """Run the guarded final grasp.

    Returns ``(status, message)`` where status is ``"waiting"`` (not ready yet),
    ``"done"`` or ``"failed"``.  Shared by the manual R key and the one-key grasp.
    """
    if not (hi_gate["passed"] and (hi_handle is not None or grip_profile is not None)
            and current_support_plane is not None and current_target is not None):
        return "waiting", "等待稳定手柄区域和当前支撑面。"
    if locked_handle is None or "grasp_tcp_z_m" not in locked_handle:
        return "failed", "未记录自适应夹持高度；请重新运行下降。"
    locked_plane_z = locked_handle.get("support_plane_z_m")
    if (locked_plane_z is None or abs(current_support_plane.z_m - locked_plane_z)
            > SUPPORT_PLANE_SHIFT_MAX_M):
        return "failed", "下降后支撑面高度发生变化；拒绝继续夹持。"
    locked_target = np.asarray(locked_handle["target_base_xyz_m"], dtype=np.float64)
    target_shift = float(np.linalg.norm(
        np.asarray(current_target, dtype=np.float64) - locked_target))
    if target_shift > SCREWDRIVER_TARGET_SHIFT_MAX_M:
        return "failed", "手柄目标在下降后移动 %.1fmm。" % (target_shift * 1000.0)
    ok = execute_screwdriver_grasp(
        robot, locked_target, float(locked_handle["grasp_tcp_z_m"]),
        current_support_plane.z_m, calibration,
        orientation=locked_handle.get("orientation", TOOL_ORIENTATION),
        grip_profile=grip_profile)
    return ("done", None) if ok else ("failed", "夹持未完成（力矩或姿态未通过）。")


def parse_args():
    p = argparse.ArgumentParser(description="文字指定工具抓取（手外粗定位->手内精定位->抓取）")
    p.add_argument("--prompt", help="English tool name, e.g. 'a screwdriver'")
    p.add_argument("--backend", choices=("clipseg_mobilesam", "yoloe"),
                   default="clipseg_mobilesam", help="local segmentation backend")
    p.add_argument("--checkpoint", help="override the selected backend's local weights")
    p.add_argument("--stage", choices=("vision", "observe", "rotate", "descent",
                                       "grasp", "place"), default="vision",
                   help="explicit validation stage; default vision sends no motion commands")
    p.add_argument("--auto", action="store_true",
                   help="for grasp/place stages, let key 'a' advance after observation")
    p.add_argument("--profile-config", type=Path,
                   default=SCRIPT_ROOT / "tool_profiles.json",
                   help="cell-specific approved tool grip parameters")
    p.add_argument("--show-angle", action="store_true",
                   help="display base-plane tool angle without enabling rotation")
    p.add_argument("--place-config", type=Path,
                   default=SCRIPT_ROOT / "fixed_placement.json",
                   help="approved fixed-placement coordinates for this cell")
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
