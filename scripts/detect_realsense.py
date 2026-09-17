"""List connected Intel RealSense devices and their USB connection type."""

from light_sam.realsense_camera import list_devices


try:
    devices = list_devices()
    if not devices:
        print("未检测到 Intel RealSense 设备。")
    else:
        for device in devices:
            print(f"型号: {device.model} | 序列号: {device.serial} | USB: {device.usb_type}")
except RuntimeError as exc:
    print(f"检测失败：{exc}")
