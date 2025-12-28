#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实单臂控制器
坐标系说明：
- flange_init: 机械臂零位时法兰盘的坐标系（与bimanual的FK/IK一致）
- flange: 当前法兰盘坐标系
- gripper: 夹爪坐标系（在flange的X轴方向偏移GRIPPER_OFFSET）
"""

import time
import json
import os
import numpy as np
from typing import Optional, List
from scipy.spatial.transform import Rotation as R
from bimanual import SingleArm


# 默认夹爪校准数据
DEFAULT_GRIPPER_CALIBRATION = {
    "-1.0": {"mean": 0.0}, "-0.5": {"mean": 0.0}, "0.0": {"mean": 0.0},
    "0.5": {"mean": 4.0}, "1.0": {"mean": 12.5}, "1.5": {"mean": 20.5},
    "2.0": {"mean": 30.0}, "2.5": {"mean": 38.5}, "3.0": {"mean": 47.5},
    "3.5": {"mean": 56.5}, "4.0": {"mean": 65.0}, "4.5": {"mean": 73.5},
    "5.0": {"mean": 82.0}
}

# 夹爪偏移（末端执行器相对于法兰盘，在法兰盘X轴方向）
GRIPPER_OFFSET = np.array([0.15, 0.0, 0.0])


class RealSingleArm:
    """
    真实单臂控制器
    
    所有位姿都在 flange_init 坐标系下表示（与bimanual的FK/IK一致）
    """
    
    def __init__(self, can_port: str = 'can0', arm_type: int = 0, 
                 calib_path: str = "gripper_calibration.json"):
        self.config = {"can_port": can_port, "type": arm_type}
        print(f"[RealSingleArm] Connecting on {can_port}...")
        self.arm = SingleArm(self.config)
        self.calib_points = self._load_calibration(calib_path)
    
    def _load_calibration(self, calib_path: str) -> List:
        """加载夹爪校准数据"""
        calib_data = DEFAULT_GRIPPER_CALIBRATION
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
            except Exception:
                pass
        points = [(v["mean"], float(k)) for k, v in calib_data.items()]
        points.sort(key=lambda x: x[0])
        return points
    
    def _real_to_set_width(self, real_mm: float) -> float:
        """真实宽度(mm) -> 设定值"""
        if not self.calib_points:
            return real_mm
        pts = self.calib_points
        if real_mm <= pts[0][0]:
            return pts[0][1]
        if real_mm >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            r1, s1 = pts[i]
            r2, s2 = pts[i + 1]
            if r1 <= real_mm <= r2 and abs(r2 - r1) > 1e-6:
                return s1 + (real_mm - r1) / (r2 - r1) * (s2 - s1)
        return pts[-1][1]
    
    # ==================== 坐标变换工具 ====================
    
    @staticmethod
    def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
        """4x4矩阵 -> [x, y, z, roll, pitch, yaw]"""
        return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_euler('xyz')])
    
    @staticmethod
    def xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
        """[x, y, z, roll, pitch, yaw] -> 4x4矩阵"""
        T = np.eye(4)
        T[:3, 3] = xyzrpy[:3]
        T[:3, :3] = R.from_euler('xyz', xyzrpy[3:]).as_matrix()
        return T
    
    @staticmethod
    def gripper_to_flange(T_ref_gripper: np.ndarray) -> np.ndarray:
        """
        夹爪位姿 -> 法兰盘位姿
        T_ref_flange = T_ref_gripper @ inv(T_flange_gripper)
        """
        T_flange_gripper = np.eye(4)
        T_flange_gripper[:3, 3] = GRIPPER_OFFSET
        return T_ref_gripper @ np.linalg.inv(T_flange_gripper)
    
    @staticmethod
    def flange_to_gripper(T_ref_flange: np.ndarray) -> np.ndarray:
        """
        法兰盘位姿 -> 夹爪位姿
        T_ref_gripper = T_ref_flange @ T_flange_gripper
        """
        T_flange_gripper = np.eye(4)
        T_flange_gripper[:3, 3] = GRIPPER_OFFSET
        return T_ref_flange @ T_flange_gripper
    
    # ==================== 位姿读取 ====================
    
    def go_home(self):
        """回零位"""
        self.arm.go_home()
    
    def get_flange_pose(self) -> np.ndarray:
        """
        获取当前法兰盘位姿 (4x4)
        返回 T_flange_init_flange（在flange_init坐标系下的当前法兰盘位姿）
        """
        xyzrpy = self.arm.get_ee_pose_xyzrpy()
        return self.xyzrpy_to_T(xyzrpy)
    
    def get_gripper_pose(self) -> np.ndarray:
        """
        获取当前夹爪位姿 (4x4)
        返回 T_flange_init_gripper（在flange_init坐标系下的当前夹爪位姿）
        """
        return self.flange_to_gripper(self.get_flange_pose())
    
    # ==================== 位姿控制 ====================
    
    def set_flange_pose(self, T_flange_init_flange: np.ndarray):
        """
        设置法兰盘目标位姿
        
        Args:
            T_flange_init_flange: 法兰盘在flange_init坐标系下的目标位姿 (4x4)
        """
        xyzrpy = self.T_to_xyzrpy(T_flange_init_flange)
        self.arm.set_ee_pose_xyzrpy(xyzrpy)
    
    def set_gripper_pose(self, T_flange_init_gripper: np.ndarray):
        """
        设置夹爪目标位姿（自动转换为法兰盘位姿）
        
        Args:
            T_flange_init_gripper: 夹爪在flange_init坐标系下的目标位姿 (4x4)
        """
        T_flange_init_flange = self.gripper_to_flange(T_flange_init_gripper)
        self.set_flange_pose(T_flange_init_flange)
    
    def set_gripper_width(self, width_mm: float):
        """设置夹爪宽度 (真实宽度mm)"""
        set_val = self._real_to_set_width(width_mm)
        self.arm.set_catch_pos(set_val)
    
    def move_to(self, T_flange_init_ee: np.ndarray, gripper_width_mm: float = 30.0, 
                is_gripper_pose: bool = True):
        """
        移动到目标位姿
        
        Args:
            T_flange_init_ee: 目标位姿 (4x4)，在flange_init坐标系下
            gripper_width_mm: 夹爪宽度 (mm)
            is_gripper_pose: True=输入是夹爪位姿，False=输入是法兰盘位姿
        """
        if is_gripper_pose:
            self.set_gripper_pose(T_flange_init_ee)
        else:
            self.set_flange_pose(T_flange_init_ee)
        self.set_gripper_width(gripper_width_mm)
    
    def execute_trajectory(self, poses: List[np.ndarray], 
                           gripper_widths: Optional[List[float]] = None,
                           is_gripper_pose: bool = True, dt: float = 0.1):
        """
        执行轨迹
        
        Args:
            poses: 位姿列表 (4x4矩阵)，在flange_init坐标系下
            gripper_widths: 夹爪宽度列表 (mm)，None则保持30mm
            is_gripper_pose: 是否为夹爪位姿
            dt: 控制周期 (秒)
        """
        n = len(poses)
        if gripper_widths is None:
            gripper_widths = [30.0] * n
        
        print(f"[RealSingleArm] Executing {n} steps...")
        
        for i, (pose, width) in enumerate(zip(poses, gripper_widths)):
            t0 = time.time()
            if not np.any(np.isnan(pose)):
                try:
                    self.move_to(pose, width, is_gripper_pose)
                except Exception as e:
                    print(f"[RealSingleArm] Step {i} error: {e}")
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print("[RealSingleArm] Trajectory finished.")


if __name__ == "__main__":
    arm = RealSingleArm(can_port='can1')
    
    print("Current flange pose (T_flange_init_flange):\n", arm.get_flange_pose())
    print("Current gripper pose (T_flange_init_gripper):\n", arm.get_gripper_pose())
    
    # 测试：在flange_init坐标系下的目标位姿
    target = np.eye(4)
    target[:3, 3] = [0.3, 0.0, 0.15]
    
    print("Moving to target...")
    arm.move_to(target, gripper_width_mm=40.0, is_gripper_pose=True)
    time.sleep(30)
    
    arm.go_home()
