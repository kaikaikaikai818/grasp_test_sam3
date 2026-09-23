"""Safe RealSense preview. Press G to infer one frame; Q or Esc exits."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from app import build_models, infer, load_config, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--serial", help="optional RealSense serial number")
    parser.add_argument("--output", type=Path, default=Path("outputs/realsense"))
    args = parser.parse_args()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    sam, grasp = build_models(load_config(args.config))
    last_overlay = None
    try:
        for _ in range(15):
            pipeline.wait_for_frames(5000)
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            image = np.asanyarray(color_frame.get_data()).copy()
            view = (last_overlay.copy() if last_overlay is not None else image.copy())
            cv2.putText(view, "G: infer once   Q/Esc: exit   VISUAL ONLY",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
            cv2.imshow("Text-guided planar grasp (no robot output)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                objects, prediction = infer(image, args.text, sam, grasp)
                folder = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                last_overlay, metadata = save_result(
                    folder, image, args.text, objects, prediction,
                    {"camera_serial": profile.get_device().get_info(rs.camera_info.serial_number),
                     "depth_scale": float(depth_scale), "depth_saved": "depth.npy"},
                )
                np.save(folder / "depth.npy", np.asanyarray(depth_frame.get_data()))
                print(
                    f"已保存视觉结果：{folder.resolve()} | "
                    f"候选 {metadata['object_count']}，"
                    f"语义通过 {metadata['semantic_accepted_count']}，"
                    f"安全抓取 {metadata['accepted_object_count']}"
                )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
