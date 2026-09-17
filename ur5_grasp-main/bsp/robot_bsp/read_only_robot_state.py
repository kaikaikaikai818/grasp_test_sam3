"""Read UR TCP state without constructing any robot control interface."""

import numpy as np
import rtde_receive


class ReadOnlyRobotState:
    """Small RTDE receive-only adapter used by the vision validation program."""

    def __init__(self, robot_ip, enabled=True):
        self.robot_ip = robot_ip
        self.interface = None
        self.error = None
        if not enabled:
            return
        try:
            self.interface = rtde_receive.RTDEReceiveInterface(robot_ip)
        except Exception as exc:
            self.error = str(exc)

    @property
    def available(self):
        return self.interface is not None

    def get_actual_tcp_pose(self):
        if self.interface is None:
            raise RuntimeError("机械臂只读状态接口未连接。")
        pose = np.asarray(self.interface.getActualTCPPose(), dtype=np.float64).reshape(-1)
        if pose.size != 6 or not np.all(np.isfinite(pose)):
            raise ValueError("机械臂返回了无效TCP位姿: %s" % pose)
        return pose

    def close(self):
        disconnect = getattr(self.interface, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass
        self.interface = None
