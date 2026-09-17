from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class DeviceInfo:
    model: str
    serial: str
    usb_type: str


@dataclass(frozen=True)
class CameraFrame:
    color_bgr: np.ndarray
    aligned_depth: np.ndarray
    depth_scale: float
    intrinsics: Any
    timestamp_ms: float
    device: DeviceInfo


def _device_info(device: Any, field: Any, default: str = "unknown") -> str:
    try:
        if device.supports(field):
            return str(device.get_info(field))
    except RuntimeError:
        pass
    return default


def list_devices() -> list[DeviceInfo]:
    devices = []
    try:
        connected = rs.context().query_devices()
    except RuntimeError as exc:
        raise RuntimeError(
            "RealSense SDK 初始化失败。请确认已安装驱动，并重新连接相机。"
        ) from exc
    for device in connected:
        devices.append(
            DeviceInfo(
                model=_device_info(device, rs.camera_info.name),
                serial=_device_info(device, rs.camera_info.serial_number),
                usb_type=_device_info(device, rs.camera_info.usb_type_descriptor),
            )
        )
    return devices


def find_required_devices() -> dict[str, DeviceInfo]:
    connected = list_devices()
    selected: dict[str, DeviceInfo] = {}
    for device in connected:
        name = device.model.upper()
        if "D455" in name:
            selected["d455"] = device
        elif "D435I" in name:
            selected["d435i"] = device

    missing = [model.upper() for model in ("d455", "d435i") if model not in selected]
    if missing:
        names = ", ".join(f"{d.model} ({d.serial})" for d in connected) or "无"
        raise RuntimeError(f"缺少 RealSense 设备：{', '.join(missing)}。当前检测到：{names}")
    for device in selected.values():
        if device.usb_type.startswith("2"):
            raise RuntimeError(f"{device.model} 当前连接为 USB {device.usb_type}，请改用 USB 3.x 接口。")
    return selected


class RealSenseCamera:
    def __init__(self, device: DeviceInfo, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.device = device
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.started = True
        for _ in range(15):
            self.pipeline.wait_for_frames(5000)

    def capture(self) -> CameraFrame:
        frames = self.pipeline.wait_for_frames(5000)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError(f"{self.device.model} 未返回完整的彩色和深度帧。")
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        return CameraFrame(
            color_bgr=np.asanyarray(color_frame.get_data()).copy(),
            aligned_depth=np.asanyarray(depth_frame.get_data()).copy(),
            depth_scale=self.depth_scale,
            intrinsics=intrinsics,
            timestamp_ms=float(color_frame.get_timestamp()),
            device=self.device,
        )

    def stop(self) -> None:
        if getattr(self, "started", False):
            self.pipeline.stop()
            self.started = False
