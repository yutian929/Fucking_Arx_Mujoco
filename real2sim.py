#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real2Sim：从真实相机坐标系下的末端执行器位姿生成仿真RGB和Mask图像
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import bimanual
from sim.mujoco_single_arm import MujocoSingleArm
from real.camera.camera_utils import load_eye_to_hand_matrix, load_camera_intrinsics

@dataclass
class Real2SimResult:
    """单帧渲染结果"""
    rgb: np.ndarray              # (H, W, 3) uint8
    mask: np.ndarray             # (H, W) uint8, 机械臂区域为255
    joint_angles: np.ndarray     # (6,) 关节角度
    T_world_flange: np.ndarray   # (4,4) 法兰盘世界位姿
    T_world_gripper: np.ndarray  # (4,4) 夹爪世界位姿
    ik_success: bool             # IK是否成功


class Real2Sim:
    """
    Real2Sim 渲染器
    
    用于从相机link坐标系下的末端执行器位姿生成仿真RGB和Mask图像。
    支持：
    - 设置相机位姿（相对于初始法兰盘）
    - 输入末端位姿（相机link坐标系）
    - 自动进行IK求解
    - 渲染RGB+Mask
    """
    
    # 夹爪相对于法兰盘的偏移（在法兰盘坐标系下，X轴方向）
    GRIPPER_OFFSET = np.array([0.15, 0.0, 0.0])
    
    def __init__(
        self,
        xml_path: str,
        T_flange_init_camlink: np.ndarray,
        width: int = 640,
        height: int = 480,
        fov: float = 43.27,
        gripper_offset: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        """
        初始化Real2Sim渲染器
        
        Args:
            xml_path: MuJoCo XML模型路径
            T_flange_init_camlink: 相机link在初始法兰盘坐标系下的4x4变换矩阵
                                   （来自手眼标定结果）
            width: 渲染图像宽度
            height: 渲染图像高度
            fov: 相机垂直FOV（度）
            gripper_offset: 夹爪相对于法兰盘的偏移，默认[0.15, 0, 0]
            verbose: 是否打印调试信息
        """
        self.xml_path = xml_path
        self.T_flange_init_camlink = np.array(T_flange_init_camlink, dtype=np.float64)
        self.width = width
        self.height = height
        self.fov = fov
        self.verbose = verbose
        
        if gripper_offset is not None:
            self.GRIPPER_OFFSET = np.array(gripper_offset, dtype=np.float64)
        
        # 初始化MuJoCo
        self.arm = MujocoSingleArm(xml_path, verbose=verbose)
        self.arm.init_renderer(width, height)
        
        # 计算相机在世界坐标系下的位姿
        self.T_world_flange_init = self.arm.get_body_pose("link6")
        self.T_world_camlink = self.T_world_flange_init @ self.T_flange_init_camlink
        
        # 设置相机
        self.arm.set_camera_pose("render_camera", self.T_world_camlink)
        self.arm.set_camera_fov("render_camera", self.fov)
        
        # 缓存上一次成功的关节角度（用于IK失败时的fallback）
        self._last_valid_joints = np.zeros(6)
        
        if self.verbose:
            print(f"[Real2Sim] Initialized")
            print(f"  T_world_camlink:\n{self.T_world_camlink}")
    
    # ==================== 坐标变换工具 ====================
    
    @staticmethod
    def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
        """4x4矩阵 -> [x, y, z, roll, pitch, yaw]"""
        pos = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
        return np.concatenate([pos, rpy])
    
    @staticmethod
    def xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
        """[x, y, z, roll, pitch, yaw] -> 4x4矩阵"""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = xyzrpy[:3]
        T[:3, :3] = R.from_euler('xyz', xyzrpy[3:], degrees=False).as_matrix()
        return T
    
    def gripper_to_flange(self, T_base_gripper: np.ndarray) -> np.ndarray:
        """将夹爪位姿转换为法兰盘位姿"""
        T_flange_gripper = np.eye(4, dtype=np.float64)
        T_flange_gripper[:3, 3] = self.GRIPPER_OFFSET
        return T_base_gripper @ np.linalg.inv(T_flange_gripper)
    
    def flange_to_gripper(self, T_base_flange: np.ndarray) -> np.ndarray:
        """将法兰盘位姿转换为夹爪位姿"""
        T_flange_gripper = np.eye(4, dtype=np.float64)
        T_flange_gripper[:3, 3] = self.GRIPPER_OFFSET
        return T_base_flange @ T_flange_gripper
    
    # ==================== 核心渲染方法 ====================
    
    def render(
        self,
        T_camlink_ee: np.ndarray,
        gripper_width: float = 0.02,
        is_gripper_pose: bool = True
    ) -> Real2SimResult:
        """
        渲染单帧
        
        Args:
            T_camlink_ee: 末端执行器在相机link坐标系下的4x4变换矩阵
                         如果 is_gripper_pose=True，这是夹爪位姿
                         如果 is_gripper_pose=False，这是法兰盘位姿
            gripper_width: 夹爪开合宽度 [0, 0.044]
            is_gripper_pose: 输入是否为夹爪位姿（需要转换到法兰盘）
        
        Returns:
            Real2SimResult 包含rgb, mask, 关节角度等信息
        """
        # 转换到世界坐标系
        T_world_ee = self.T_world_camlink @ T_camlink_ee
        
        # 如果输入是夹爪位姿，转换为法兰盘位姿
        if is_gripper_pose:
            T_world_flange_target = self.gripper_to_flange(T_world_ee)
        else:
            T_world_flange_target = T_world_ee
        
        # 转换到初始法兰盘坐标系（IK求解需要）
        T_flange_init_inv = np.linalg.inv(self.T_world_flange_init)
        T_flange_init_flange_target = T_flange_init_inv @ T_world_flange_target
        
        # IK求解
        xyzrpy_target = self.T_to_xyzrpy(T_flange_init_flange_target)
        
        try:
            joint_angles = bimanual.inverse_kinematics(xyzrpy_target)
            if joint_angles is None:
                ik_success = False
                joint_angles = self._last_valid_joints.copy()
            else:
                ik_success = True
                self._last_valid_joints = joint_angles.copy()
        except Exception as e:
            if self.verbose:
                print(f"[WARN] IK failed: {e}")
            ik_success = False
            joint_angles = self._last_valid_joints.copy()
        
        # 设置关节角度
        full_joints = np.zeros(8)
        full_joints[:6] = joint_angles
        full_joints[6:8] = np.clip(gripper_width, 0.0, 0.044)
        
        self.arm.set_joint_angles(full_joints)
        self.arm.forward()
        
        # 获取实际位姿
        T_world_flange = self.arm.get_body_pose("link6")
        T_world_gripper = self.flange_to_gripper(T_world_flange)
        
        # 设置目标marker用于调试
        self.arm.set_target_marker(T_world_ee)
        self.arm.forward()
        
        # 渲染
        rgb, mask = self.arm.render("render_camera", self.width, self.height, with_mask=True)
        
        return Real2SimResult(
            rgb=rgb,
            mask=mask,
            joint_angles=joint_angles,
            T_world_flange=T_world_flange,
            T_world_gripper=T_world_gripper,
            ik_success=ik_success
        )
    
    def render_batch(
        self,
        T_camlink_ee_list: List[np.ndarray],
        gripper_widths: Optional[List[float]] = None,
        is_gripper_pose: bool = True,
        show_progress: bool = True
    ) -> List[Real2SimResult]:
        """
        批量渲染
        
        Args:
            T_camlink_ee_list: 末端执行器位姿列表，每个元素是4x4变换矩阵
            gripper_widths: 夹爪宽度列表，None则全部使用0.02
            is_gripper_pose: 输入是否为夹爪位姿
            show_progress: 是否显示进度
        
        Returns:
            Real2SimResult列表
        """
        n = len(T_camlink_ee_list)
        
        if gripper_widths is None:
            gripper_widths = [0.02] * n
        
        assert len(gripper_widths) == n, "gripper_widths长度必须与位姿列表相同"
        
        results = []
        ik_fail_count = 0
        
        for i, (T_ee, gw) in enumerate(zip(T_camlink_ee_list, gripper_widths)):
            result = self.render(T_ee, gw, is_gripper_pose)
            results.append(result)
            
            if not result.ik_success:
                ik_fail_count += 1
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"[Real2Sim] Rendered {i+1}/{n} frames, IK fails: {ik_fail_count}")
        
        if show_progress:
            print(f"[Real2Sim] Done. Total: {n}, IK fails: {ik_fail_count}")
        
        return results
    
    # ==================== 辅助方法 ====================
    
    def get_arrays(self, results: List[Real2SimResult]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从渲染结果中提取RGB和Mask数组
        
        Returns:
            rgb_array: (N, H, W, 3) uint8
            mask_array: (N, H, W) uint8
        """
        rgb_list = [r.rgb for r in results]
        mask_list = [r.mask for r in results]
        return np.stack(rgb_list), np.stack(mask_list)
    
    def start_viewer(self):
        """启动MuJoCo Viewer"""
        self.arm.start_viewer()
    
    def spin(self, rate_hz: float = 60.0):
        """持续更新Viewer直到关闭"""
        self.arm.spin(rate_hz)
    
    def close(self):
        """关闭渲染器"""
        self.arm.stop_viewer()


# ==================== 测试 ====================

if __name__ == "__main__":
    import cv2

    # 参数
    XML_PATH = "SDK/R5a/meshes/R5a_R5master.xml"
    EYE_TO_HAND_PATH = "real/camera/eye_to_hand_result_left_latest.json"
    CAMERA_INTRINSICS_PATH = "real/camera/camera_intrinsics_d435i.json"
    
    # 加载标定数据
    T_flange_init_camlink = load_eye_to_hand_matrix(EYE_TO_HAND_PATH)
    _, K, dist, v_fov = load_camera_intrinsics(CAMERA_INTRINSICS_PATH)
    
    print(f"[INFO] T_flange_init_camlink:\n{T_flange_init_camlink}")
    print(f"[INFO] v_fov: {v_fov}")
    
    # 创建渲染器
    r2s = Real2Sim(
        xml_path=XML_PATH,
        T_flange_init_camlink=T_flange_init_camlink,
        width=640,
        height=480,
        fov=v_fov,
        verbose=True
    )
    
    # 生成测试序列：末端沿相机link的X轴（前方）直线移动
    # 姿态保持不变（单位矩阵）
    T_camlink_ee_list = []
    start_x = 0.5   # 起始距离 0.2m
    end_x = 0.8     # 结束距离 0.5m
    num_frames = 30
    
    for i in range(num_frames):
        T = np.eye(4, dtype=np.float64)
        # 位置：沿X轴（前方）线性移动
        x = start_x + (end_x - start_x) * i / (num_frames - 1)
        T[:3, 3] = [x, 0.0, 0.0]
        # 姿态：保持不变（单位矩阵）
        T_camlink_ee_list.append(T)
    
    print(f"[INFO] 生成 {num_frames} 帧测试序列，末端从 x={start_x}m 移动到 x={end_x}m")
    
    # 批量渲染
    results = r2s.render_batch(T_camlink_ee_list, show_progress=True)
    
    # 显示结果
    for i, res in enumerate(results):
        vis = cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR)
        
        # 叠加mask
        overlay = vis.astype(np.float32)
        overlay[res.mask > 0] = overlay[res.mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
        
        # 显示IK状态
        status = "IK OK" if res.ik_success else "IK FAIL"
        cv2.putText(overlay, f"Frame {i}: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Real2Sim", overlay.astype(np.uint8))
        key = cv2.waitKey(100)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # 启动viewer
    r2s.spin()
