#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo 单臂可视化脚本
使用 MuJoCo 3.4.0 读取XML文件，设定关节角度并可视化
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R


class MujocoSingleArm:
    """
    MuJoCo 单臂模型封装类
    支持加载模型、设置关节角度、正向运动学计算、相机渲染和可视化
    """
    
    def __init__(self, xml_path: str, verbose: bool = True):
        self.xml_path = xml_path
        self.viewer = None
        self.renderer = None
        self.render_width = 640
        self.render_height = 480
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.data.qpos[:] = 0
        self.forward()

        if verbose:
            self._print_model_info()
            self.print_body_positions()
    
    def _print_model_info(self):
        print(f"[INFO] Model: {self.xml_path}")
        print(f"  nq={self.model.nq}, njnt={self.model.njnt}, nbody={self.model.nbody}")
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jtype = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[self.model.jnt_type[i]]
            print(f"    Joint {i}: {name} ({jtype})")
    
    # ==================== 关节操作 ====================
    
    def set_joint_angles(self, angles: np.ndarray):
        angles = np.asarray(angles, dtype=np.float64)
        assert len(angles) == self.model.nq, f"Expected {self.model.nq} angles, got {len(angles)}"
        self.data.qpos[:] = angles
    
    def get_joint_angles(self) -> np.ndarray:
        return self.data.qpos.copy()
    
    def forward(self):
        mujoco.mj_forward(self.model, self.data)
    
    # ==================== Body位姿查询 ====================
    
    def get_body_pose(self, body_name: str) -> np.ndarray:
        """获取body的4x4齐次变换矩阵"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]
        return T
    
    def print_body_positions(self):
        print("\n[INFO] Body positions:")
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            pos = self.data.xpos[i]
            print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    
    # ==================== Target Marker ====================
    
    def set_target_marker(self, pos: np.ndarray, marker_name: str = "target_marker"):
        """
        设置目标标记点的位置（用于debug可视化）
        
        Args:
            pos: 位置 [x, y, z] 或 4x4变换矩阵
            marker_name: mocap body名称（需在XML中预定义为mocap="true"）
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
        if body_id == -1:
            print(f"[WARN] Marker body '{marker_name}' not found in model.")
            return
        
        # 获取mocap id
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id == -1:
            print(f"[WARN] Body '{marker_name}' is not a mocap body.")
            return
        
        # 支持多种输入格式
        if isinstance(pos, np.ndarray):
            if pos.shape == (4, 4):
                # 4x4变换矩阵
                self.data.mocap_pos[mocap_id] = pos[:3, 3]
                # 可选：也设置姿态
                rot_mat = pos[:3, :3]
                quat_scipy = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
                self.data.mocap_quat[mocap_id] = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
            elif pos.shape == (3,):
                # [x, y, z]
                self.data.mocap_pos[mocap_id] = pos
            elif pos.shape == (6,):
                # [x, y, z, roll, pitch, yaw]
                self.data.mocap_pos[mocap_id] = pos[:3]
                quat_scipy = R.from_euler('xyz', pos[3:], degrees=False).as_quat()
                self.data.mocap_quat[mocap_id] = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
            else:
                print(f"[WARN] Invalid pos format: {pos.shape}")
        else:
            # 假设是list或tuple
            pos = np.array(pos)
            self.data.mocap_pos[mocap_id] = pos[:3]
    
    # ==================== 相机渲染 ====================
    
    def init_renderer(self, width: int = 640, height: int = 480):
        self.render_width = width
        self.render_height = height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
    
    def set_camera_pose(self, cam_name: str, T_world_camlink: np.ndarray):
        """
        设置相机位姿（Link坐标系：X朝前, Y朝左, Z朝上）
        
        Args:
            cam_name: 相机名称（需在XML中预定义）
            T_world_camlink: 相机在世界坐标系下的4x4变换矩阵
                            或者 (6,) 的 [x, y, z, roll, pitch, yaw]
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id == -1:
            print(f"[WARN] Camera '{cam_name}' not found in model.")
            return
        
        # 支持两种输入格式
        if T_world_camlink.shape == (6,):
            # [x, y, z, roll, pitch, yaw] 格式
            pos = T_world_camlink[:3]
            r_input = R.from_euler('xyz', T_world_camlink[3:], degrees=False)
            mat_input = r_input.as_matrix()
        elif T_world_camlink.shape == (4, 4):
            # 4x4矩阵格式
            pos = T_world_camlink[:3, 3]
            mat_input = T_world_camlink[:3, :3]
        else:
            print(f"[WARN] Invalid pose format: {T_world_camlink.shape}")
            return
        
        # 设置位置
        self.model.cam_pos[cam_id] = pos
        
        # Link -> MuJoCo Camera 转换
        # 参考代码中的转换矩阵（从Link到MuJoCo相机坐标系）
        # MuJoCo相机: -Z朝前(看向的方向), Y朝上, X朝右
        # Link: X朝前, Y朝左, Z朝上
        # 这个矩阵将Link坐标系的点转换到MuJoCo相机坐标系
        mat_link_to_mjcam = np.array([
            [ 0,  0, -1],  # mjcam X = -link Z? 不对，重新推导
            [-1,  0,  0],  # mjcam Y = -link X? 
            [ 0,  1,  0],  # mjcam Z = link Y?
        ], dtype=np.float64)
        
        # 应用转换: R_world_mjcam = R_world_link @ R_link_mjcam
        mat_final = mat_input @ mat_link_to_mjcam
        
        r_final = R.from_matrix(mat_final)
        quat_scipy = r_final.as_quat()  # [x, y, z, w]
        
        # 转换为 MuJoCo 顺序 [w, x, y, z]
        self.model.cam_quat[cam_id] = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]

    def set_camera_fov(self, cam_name: str, fov_deg: float):
        """设置相机垂直FOV（度）"""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id != -1:
            self.model.cam_fovy[cam_id] = fov_deg

    def render(self, cam_name: str, width: int = 640, height: int = 480,
               with_mask: bool = False, exclude_geom_names: list = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        渲染RGB图像，可选同时返回分割掩码
        
        Args:
            cam_name: 相机名称
            width: 图像宽度
            height: 图像高度
            with_mask: 是否同时返回掩码
            exclude_geom_names: 要排除的geom名称列表（掩码中不包含这些）
        
        Returns:
            如果 with_mask=False: (rgb, None)
            如果 with_mask=True:  (rgb, mask)
            - rgb: RGB图像 (H, W, 3) uint8
            - mask: 二值掩码 (H, W) uint8, 机械臂区域为255
        """
        if self.renderer is None or self.render_width != width or self.render_height != height:
            self.init_renderer(width, height)
        
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        
        # 渲染RGB
        if cam_id != -1:
            self.renderer.update_scene(self.data, camera=cam_name)
        else:
            self.renderer.update_scene(self.data)
        rgb = self.renderer.render().copy()
        
        # 如果不需要掩码，直接返回
        if not with_mask:
            return rgb, None
        
        # 渲染分割掩码
        self.renderer.enable_segmentation_rendering()
        if cam_id != -1:
            self.renderer.update_scene(self.data, camera=cam_name)
        else:
            self.renderer.update_scene(self.data)
        seg = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        
        # 处理分割结果
        # seg[:,:,0] 是 geom id
        geom_ids = seg[:, :, 0].astype(np.int32)
        
        # 默认排除的geom
        if exclude_geom_names is None:
            exclude_geom_names = ['floor', 'stand']
        
        # 获取要排除的geom id
        exclude_ids = set([-1])  # 背景
        for name in exclude_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                exclude_ids.add(gid)
        
        # 创建掩码：所有非排除的geom都算作机械臂
        mask = np.zeros((height, width), dtype=np.uint8)
        for gid in np.unique(geom_ids):
            if gid not in exclude_ids and gid >= 0:
                mask[geom_ids == gid] = 255
        
        return rgb, mask
    
    # ==================== Viewer ====================
    
    def start_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def stop_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def is_viewer_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()
    
    def update(self):
        self.forward()
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
    
    def spin(self, rate_hz: float = 60.0):
        if self.viewer is None:
            self.start_viewer()
        
        dt = 1.0 / rate_hz
        while self.is_viewer_running():
            self.update()
            time.sleep(dt)
        self.viewer = None
    
    def __del__(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except: pass
        if self.renderer is not None:
            try: self.renderer.close()
            except: pass



if __name__ == "__main__":
    import cv2
    
    xml_path = "/home/yutian/Hand2Gripper_phantom/submodules/Fucking_Arx_Mujoco/SDK/R5a/meshes/R5a_R5master.xml"
    
    arm = MujocoSingleArm(xml_path)
    arm.set_joint_angles([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02])
    arm.forward()
    arm.print_body_positions()
    
    # 测试相机渲染
    cam_pose = np.array([0.5, 0.0, 0.2, 0.0, 0.0, np.pi])
    arm.set_camera_pose("render_camera", cam_pose)
    arm.set_camera_fov("render_camera", 60.0)
    arm.forward()
    
    # 渲染RGB和掩码
    rgb, mask = arm.render("render_camera", 640, 480, with_mask=True)
    
    print(f"[INFO] RGB shape: {rgb.shape}, Mask shape: {mask.shape}")
    print(f"[INFO] Mask unique values: {np.unique(mask)}")
    
    # 显示
    cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Mask", mask)
    
    # 叠加显示
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    cv2.imshow("Overlay", overlay.astype(np.uint8))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 测试俯视
    cam_pose_top = np.array([0.1, 0.0, 0.8, 0.0, np.pi/2, 0.0])
    arm.set_camera_pose("render_camera", cam_pose_top)
    arm.forward()
    
    rgb_top, mask_top = arm.render("render_camera", 640, 480, with_mask=True)
    cv2.imshow("Top RGB", cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))
    cv2.imshow("Top Mask", mask_top)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    arm.spin()