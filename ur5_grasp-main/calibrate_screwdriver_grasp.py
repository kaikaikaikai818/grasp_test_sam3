#!/usr/bin/env python
"""One-time, read-only depth calibration for the flat-cardboard screwdriver test.

This script never calls moveL or grip.  The operator moves the robot manually
with the teach pendant for contact calibration.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot
from bsp.robot_bsp.read_only_robot_state import ReadOnlyRobotState
from bsp.camera_bsp.sam_tool_detect import SamToolDetector, TemporalResultFilter
from bsp.camera_bsp.screwdriver_grasp import (find_screwdriver_handle,
                                               fit_horizontal_support_plane,
                                               estimate_handle_thickness,
                                               result_at_handle, save_calibration)

ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "grasp_surface_calibration.json"
ROBOT_IP = "192.168.1.35"
HI_SERIAL = "215222074676"
CAM_INI = str(ROOT / "camera_20260906.ini")
CAM2END_PATH = str(ROOT / "cam2end_20260906.txt")
MAX_PLANE_SHIFT_M = 0.008


def read_partial_calibration():
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}


def main():
    parser = argparse.ArgumentParser(description="只读标定纸箱平面与螺丝刀夹持高度；不发送运动或夹爪命令。")
    parser.add_argument("mode", choices=("plane", "contact"), help="plane: 空纸箱；contact: 手动示教夹持高度")
    parser.add_argument("--prompt", default="a screwdriver")
    args = parser.parse_args()

    robot = UR_Robot(robot_ip=ROBOT_IP, is_use_robot=False, is_use_camera=True,
                     connect_robot=False, is_use_gripper=False, camera_serial=HI_SERIAL,
                     cam2end_path=CAM2END_PATH, cam_ini_path=CAM_INI)
    robot_state = ReadOnlyRobotState(ROBOT_IP, enabled=True)
    detector = SamToolDetector(args.prompt) if args.mode == "contact" else None
    tracker = TemporalResultFilter(stable_frames=3) if detector else None
    window = "screwdriver calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("[只读标定] 本脚本不会发送机械臂运动或夹爪命令。")
    if args.mode == "plane":
        print("移开螺丝刀，让 D435i 看见空支撑面（桌面/底板）；点击画面后按小写 s 保存。q 退出。")
    else:
        print("放回螺丝刀，用示教器将张开的夹爪手动放到手柄中段的目标夹持高度。")
        print("画面显示 HANDLE STABLE 后，点击画面按小写 k 保存 TCP 偏移。q 退出。")

    try:
        while True:
            color, depth = robot.get_camera_data()
            if color is None:
                continue
            image = color.copy()
            try:
                tcp = np.asarray(robot_state.get_actual_tcp_pose(), dtype=float).reshape(6)
            except Exception:
                tcp = None

            def pixel_to_base(u, v, z_m):
                if tcp is None:
                    raise ValueError("read-only TCP unavailable")
                p_cam = robot.pixel_to_camera(u, v, z_m * 1000.0)
                _, p_base = robot.camera_to_base(p_cam, tcp_pose=tcp)
                return np.asarray(p_base, dtype=float) / 1000.0

            result = handle = selected = None
            status = ""
            if args.mode == "plane":
                plane, reason = fit_horizontal_support_plane(depth, robot.camera.scale, pixel_to_base)
                status = reason if plane is None else "PLANE z=%.4fm spread=%.1fmm" % (
                    plane.z_m, plane.spread_m * 1000.0)
            else:
                raw = detector.detect(color, depth, robot.camera.scale)
                result, tracking = tracker.update(raw)
                if result is not None:
                    handle, reason = find_screwdriver_handle(result, depth, robot.camera.scale)
                    if handle is not None:
                        selected = result_at_handle(result, handle)
                        image = detector.draw(image, selected)
                        cv2.circle(image, handle.center_px, 12, (255, 255, 0), 2)
                        diameter_est, _ = estimate_handle_thickness(
                            handle.radius_px, handle.depth_m,
                            float(robot.cam_intrinsics[0, 0]))
                        status = "HANDLE %s  top=%.1fmm median=%.1fmm  D~%.1fmm" % (
                            tracking, handle.depth_m * 1000.0, handle.median_depth_m * 1000.0,
                            (diameter_est * 1000.0) if diameter_est is not None else -1.0)
                    else:
                        status = "HANDLE REJECTED: " + reason
                else:
                    status = "SEARCHING screwdriver"
                plane, plane_reason = fit_horizontal_support_plane(
                    depth, robot.camera.scale, pixel_to_base,
                    exclude_mask=(selected or {}).get("mask"))
                if plane is None:
                    status += " | plane: " + plane_reason

            cv2.rectangle(image, (0, 0), (image.shape[1], 62), (20, 20, 20), -1)
            cv2.putText(image, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 0) if "REJECTED" not in status else (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "s: save plane" if args.mode == "plane" else "k: save contact TCP", (10, 51),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.mode == "plane" and key == ord("s"):
                if plane is None:
                    print("[拒绝保存] " + reason)
                    continue
                data = read_partial_calibration()
                data.update({"support_plane_z_m": plane.z_m, "support_plane_spread_m": plane.spread_m,
                             "support_plane_inlier_count": plane.inlier_count,
                             "plane_calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
                save_calibration(CALIBRATION_PATH, data)
                print("[已保存] 支撑面 z=%.4fm -> %s" % (plane.z_m, CALIBRATION_PATH))
            if args.mode == "contact" and key == ord("k"):
                if tcp is None or handle is None or selected is None or plane is None:
                    print("[拒绝保存] 等待稳定手柄、支撑平面和只读 TCP。")
                    continue
                data = read_partial_calibration()
                if "support_plane_z_m" not in data:
                    print("[拒绝保存] 请先运行 plane 标定。")
                    continue
                target = pixel_to_base(*handle.center_px, handle.depth_m)
                focal_px = float(robot.cam_intrinsics[0, 0])
                thickness, thickness_reason = estimate_handle_thickness(
                    handle.radius_px, handle.depth_m, focal_px)
                tcp_clearance = float(tcp[2] - plane.z_m)
                offset = float(tcp[2] - target[2])
                if thickness is None:
                    print("[拒绝保存] 无法估算手柄直径：%s" % thickness_reason)
                    continue
                if not (0.010 <= thickness <= 0.100):
                    print("[拒绝保存] 估算手柄直径 %.1fmm 不合理（应在 10~100mm），"
                          "请让 D435i 完整看到手柄。" % (thickness * 1000.0))
                    continue
                gripper_offset = float(tcp[2] - (plane.z_m + thickness / 2.0))
                if tcp_clearance <= 0.010:
                    print("[拒绝保存] TCP 离支撑面不足 10mm。")
                    continue
                data.update({"contact_offset_m": offset,
                             "minimum_tcp_plane_clearance_m": tcp_clearance - 0.003,
                             "handle_surface_above_plane_m": thickness,
                             "gripper_offset_m": gripper_offset,
                             "handle_radius_px": float(handle.radius_px),
                             "contact_calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                             "prompt": args.prompt})
                save_calibration(CALIBRATION_PATH, data)
                print("[已保存] 估算手柄直径=%.1fmm，夹爪偏移=%.1fmm，TCP-手柄顶面=%.1fmm，"
                      "最小TCP平面净空=%.1fmm" %
                      (thickness * 1000.0, gripper_offset * 1000.0, offset * 1000.0,
                       (tcp_clearance - 0.003) * 1000.0))
    finally:
        if getattr(robot, "camera", None) is not None:
            robot.camera.stop()
        robot_state.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
