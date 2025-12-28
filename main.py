import os
import cv2
import json
import numpy as np
from typing import List, Optional
from scipy.spatial.transform import Rotation as R
from real.camera.aruco_utils import detect_aruco, draw_aruco, ArucoResult
from real.camera.camera_utils import load_camera_intrinsics, load_eye_to_hand_matrix, T_optical_to_link
from sim.mujoco_single_arm import MujocoSingleArm
import bimanual


# =========================
# 夹爪偏移常量
# =========================
# 夹爪相对于法兰盘(link6)的偏移，在法兰盘坐标系下
# 夹爪在法兰盘X轴方向延伸15cm
GRIPPER_OFFSET_IN_FLANGE = np.array([0.15, 0.0, 0.0])  # [x, y, z] 单位：米


def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """将4x4变换矩阵转换为 [x, y, z, roll, pitch, yaw]"""
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    rpy = rot.as_euler('xyz', degrees=False)
    return np.concatenate([pos, rpy])


def xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
    """将 [x, y, z, roll, pitch, yaw] 转换为4x4变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyzrpy[:3]
    T[:3, :3] = R.from_euler('xyz', xyzrpy[3:], degrees=False).as_matrix()
    return T


def gripper_to_flange(T_base_gripper: np.ndarray) -> np.ndarray:
    """
    将夹爪目标位姿转换为法兰盘目标位姿
    
    T_base_flange = T_base_gripper @ inv(T_flange_gripper)
    
    其中 T_flange_gripper 是夹爪在法兰盘坐标系下的位姿（只有平移，无旋转）
    """
    # T_flange_gripper: 夹爪相对于法兰盘的变换（只有平移）
    T_flange_gripper = np.eye(4, dtype=np.float64)
    T_flange_gripper[:3, 3] = GRIPPER_OFFSET_IN_FLANGE
    
    # T_base_flange = T_base_gripper @ inv(T_flange_gripper)
    T_gripper_flange = np.linalg.inv(T_flange_gripper)
    T_base_flange = T_base_gripper @ T_gripper_flange
    
    return T_base_flange


def flange_to_gripper(T_base_flange: np.ndarray) -> np.ndarray:
    """
    将法兰盘位姿转换为夹爪位姿
    
    T_base_gripper = T_base_flange @ T_flange_gripper
    """
    T_flange_gripper = np.eye(4, dtype=np.float64)
    T_flange_gripper[:3, 3] = GRIPPER_OFFSET_IN_FLANGE
    
    T_base_gripper = T_base_flange @ T_flange_gripper
    return T_base_gripper


if __name__ == "__main__":
    # =========================
    # 参数设定
    # =========================
    IMAGE_PATH = "assert/calib_screenshot_raw_20251228_091850.png"
    CAMERA_INTRINSICS_PATH = "real/camera/camera_intrinsics_d435i.json"
    EYE_TO_HAND_LEFT_PATH = "real/camera/eye_to_hand_result_left_latest.json"
    EYE_TO_HAND_RIGHT_PATH = "real/camera/eye_to_hand_result_right_latest.json"
    XML_PATH = "SDK/R5a/meshes/R5a_R5master.xml"
    
    MARKER_SIZE = 0.05
    TARGET_MARKER_ID = 0
    
    # =========================
    # 加载相机内参和手眼标定
    # =========================
    _, K, dist, v_fov = load_camera_intrinsics(CAMERA_INTRINSICS_PATH)
    T_flange_init_L_camlink = load_eye_to_hand_matrix(EYE_TO_HAND_LEFT_PATH)
    T_flange_init_R_camlink = load_eye_to_hand_matrix(EYE_TO_HAND_RIGHT_PATH)
    
    print(f"[INFO] 相机内参:\n{K}")
    print(f"[INFO] 夹爪偏移 (法兰盘坐标系): {GRIPPER_OFFSET_IN_FLANGE} m")
    print(f"\n[INFO] T_flange_init_L_camlink:\n{T_flange_init_L_camlink}")
    print(f"\n[INFO] T_flange_init_R_camlink:\n{T_flange_init_R_camlink}")
    
    # =========================
    # 读取图片并检测ArUco
    # =========================
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] 图片不存在: {IMAGE_PATH}")
        exit(1)
    
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] 无法读取图片: {IMAGE_PATH}")
        exit(1)
    
    print(f"\n[INFO] 读取图片: {IMAGE_PATH}, 尺寸: {img.shape}")
    
    results = detect_aruco(img, K, dist, MARKER_SIZE, target_id=TARGET_MARKER_ID)
    
    if not results:
        print("[ERROR] 未检测到ArUco码")
        exit(1)
    
    target_aruco = results[0]
    print(f"\n[INFO] 检测到ArUco ID={target_aruco.marker_id}")
    print(f"  T_optical_marker:\n{target_aruco.T_cam_marker}")

    # =========================
    # 可视化ArUco检测结果
    # =========================
    vis = draw_aruco(img, results, K, dist)
    cv2.imshow("ArUco Detection", vis)
    print("\n[INFO] 按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # =========================
    # 坐标变换：optical -> link -> flange_init
    # =========================
    T_optical_marker = target_aruco.T_cam_marker
    T_link_optical = T_optical_to_link()
    T_camlink_marker = T_link_optical @ T_optical_marker
    
    print(f"\n[INFO] T_camlink_marker (marker在相机link坐标系下):\n{T_camlink_marker}")
    
    # Marker在初始法兰坐标系下的位姿（这是夹爪的目标位姿）
    T_flange_init_L_marker = T_flange_init_L_camlink @ T_camlink_marker
    T_flange_init_R_marker = T_flange_init_R_camlink @ T_camlink_marker
    
    print(f"\n[INFO] Marker在左臂初始法兰坐标系下的位姿 (夹爪目标):")
    xyzrpy_gripper_target_L = T_to_xyzrpy(T_flange_init_L_marker)
    print(f"  xyz:  [{xyzrpy_gripper_target_L[0]:.4f}, {xyzrpy_gripper_target_L[1]:.4f}, {xyzrpy_gripper_target_L[2]:.4f}] m")
    print(f"  rpy:  [{xyzrpy_gripper_target_L[3]:.4f}, {xyzrpy_gripper_target_L[4]:.4f}, {xyzrpy_gripper_target_L[5]:.4f}] rad")
    
    print(f"\n[INFO] Marker在右臂初始法兰坐标系下的位姿 (夹爪目标):")
    xyzrpy_gripper_target_R = T_to_xyzrpy(T_flange_init_R_marker)
    print(f"  xyz:  [{xyzrpy_gripper_target_R[0]:.4f}, {xyzrpy_gripper_target_R[1]:.4f}, {xyzrpy_gripper_target_R[2]:.4f}] m")
    print(f"  rpy:  [{xyzrpy_gripper_target_R[3]:.4f}, {xyzrpy_gripper_target_R[4]:.4f}, {xyzrpy_gripper_target_R[5]:.4f}] rad")
    
    # =========================
    # 将夹爪目标位姿转换为法兰盘目标位姿
    # =========================
    T_flange_init_L_flange_target = gripper_to_flange(T_flange_init_L_marker)
    T_flange_init_R_flange_target = gripper_to_flange(T_flange_init_R_marker)
    
    xyzrpy_flange_target_L = T_to_xyzrpy(T_flange_init_L_flange_target)
    xyzrpy_flange_target_R = T_to_xyzrpy(T_flange_init_R_flange_target)
    
    print(f"\n[INFO] 左臂法兰盘目标位姿 (用于IK):")
    print(f"  xyz:  [{xyzrpy_flange_target_L[0]:.4f}, {xyzrpy_flange_target_L[1]:.4f}, {xyzrpy_flange_target_L[2]:.4f}] m")
    print(f"  rpy:  [{xyzrpy_flange_target_L[3]:.4f}, {xyzrpy_flange_target_L[4]:.4f}, {xyzrpy_flange_target_L[5]:.4f}] rad")
    
    print(f"\n[INFO] 右臂法兰盘目标位姿 (用于IK):")
    print(f"  xyz:  [{xyzrpy_flange_target_R[0]:.4f}, {xyzrpy_flange_target_R[1]:.4f}, {xyzrpy_flange_target_R[2]:.4f}] m")
    print(f"  rpy:  [{xyzrpy_flange_target_R[3]:.4f}, {xyzrpy_flange_target_R[4]:.4f}, {xyzrpy_flange_target_R[5]:.4f}] rad")
    
    # =========================
    # 逆运动学求解（使用法兰盘目标位姿）
    # =========================
    print("\n" + "="*50)
    print("[INFO] 左臂逆运动学求解...")
    print("="*50)
    try:
        ik_joint_angles_L = bimanual.inverse_kinematics(xyzrpy_flange_target_L)
        if ik_joint_angles_L is None:
            print("[WARN] 左臂逆运动学求解失败")
            ik_joint_angles_L = None
        else:
            print(f"[INFO] 左臂关节角度: {np.round(ik_joint_angles_L, 4)}")
            fk_result_L = bimanual.forward_kinematics(ik_joint_angles_L)
            print(f"[INFO] 左臂正运动学验证 (法兰盘): {np.round(fk_result_L, 4)}")
            print(f"[INFO] 左臂法兰盘位置误差: {np.linalg.norm(fk_result_L[:3] - xyzrpy_flange_target_L[:3]):.6f} m")
    except Exception as e:
        print(f"[WARN] 左臂逆运动学异常: {e}")
        ik_joint_angles_L = None
    
    print("\n" + "="*50)
    print("[INFO] 右臂逆运动学求解...")
    print("="*50)
    try:
        ik_joint_angles_R = bimanual.inverse_kinematics(xyzrpy_flange_target_R)
        if ik_joint_angles_R is None:
            print("[WARN] 右臂逆运动学求解失败")
            ik_joint_angles_R = None
        else:
            print(f"[INFO] 右臂关节角度: {np.round(ik_joint_angles_R, 4)}")
            fk_result_R = bimanual.forward_kinematics(ik_joint_angles_R)
            print(f"[INFO] 右臂正运动学验证 (法兰盘): {np.round(fk_result_R, 4)}")
            print(f"[INFO] 右臂法兰盘位置误差: {np.linalg.norm(fk_result_R[:3] - xyzrpy_flange_target_R[:3]):.6f} m")
    except Exception as e:
        print(f"[WARN] 右臂逆运动学异常: {e}")
        ik_joint_angles_R = None
    
    # =========================
    # MuJoCo正运动学验证
    # =========================
    print("\n" + "="*50)
    print("[INFO] 在MuJoCo中验证...")
    print("="*50)
    
    mujoco_left_arm = MujocoSingleArm(XML_PATH, verbose=False)
    T_world_flange_init_L = mujoco_left_arm.get_body_pose("link6")
    T_world_cameralink = T_world_flange_init_L @ T_flange_init_L_camlink
    
    # 验证左臂
    if ik_joint_angles_L is not None:
        print("\n[INFO] 验证左臂...")
        full_ik_joint_angles_L = np.zeros(8)
        full_ik_joint_angles_L[:6] = ik_joint_angles_L
        full_ik_joint_angles_L[6:8] = [0.02, 0.02]
        
        mujoco_left_arm.set_joint_angles(full_ik_joint_angles_L)
        mujoco_left_arm.forward()

        mujoco_left_arm.set_camera_pose("render_camera", T_world_cameralink)
        mujoco_left_arm.set_camera_fov("render_camera", v_fov)
        
        # 设置目标marker位置（用于debug）- 这是夹爪应该到达的位置
        T_world_marker_L = T_world_cameralink @ T_camlink_marker
        mujoco_left_arm.set_target_marker(T_world_marker_L)
        
        mujoco_left_arm.forward()
        
        # 获取法兰盘位姿
        T_world_flange_L = mujoco_left_arm.get_body_pose("link6")
        xyzrpy_world_flange_L = T_to_xyzrpy(T_world_flange_L)
        
        # 计算夹爪位姿
        T_world_gripper_L = flange_to_gripper(T_world_flange_L)
        xyzrpy_world_gripper_L = T_to_xyzrpy(T_world_gripper_L)
        
        print(f"  MuJoCo 法兰盘(link6)位姿:")
        print(f"    xyz:  [{xyzrpy_world_flange_L[0]:.4f}, {xyzrpy_world_flange_L[1]:.4f}, {xyzrpy_world_flange_L[2]:.4f}] m")
        print(f"    rpy:  [{xyzrpy_world_flange_L[3]:.4f}, {xyzrpy_world_flange_L[4]:.4f}, {xyzrpy_world_flange_L[5]:.4f}] rad")
        
        print(f"  MuJoCo 夹爪位姿 (法兰盘+偏移):")
        print(f"    xyz:  [{xyzrpy_world_gripper_L[0]:.4f}, {xyzrpy_world_gripper_L[1]:.4f}, {xyzrpy_world_gripper_L[2]:.4f}] m")
        print(f"    rpy:  [{xyzrpy_world_gripper_L[3]:.4f}, {xyzrpy_world_gripper_L[4]:.4f}, {xyzrpy_world_gripper_L[5]:.4f}] rad")
        
        print(f"  目标位置 (marker):")
        xyzrpy_world_marker_L = T_to_xyzrpy(T_world_marker_L)
        print(f"    xyz:  [{xyzrpy_world_marker_L[0]:.4f}, {xyzrpy_world_marker_L[1]:.4f}, {xyzrpy_world_marker_L[2]:.4f}] m")
        print(f"    rpy:  [{xyzrpy_world_marker_L[3]:.4f}, {xyzrpy_world_marker_L[4]:.4f}, {xyzrpy_world_marker_L[5]:.4f}] rad")
        
        # 计算夹爪到目标的误差
        error_xyz_gripper = np.linalg.norm(xyzrpy_world_gripper_L[:3] - xyzrpy_world_marker_L[:3])
        print(f"  夹爪位置误差: {error_xyz_gripper:.6f} m")
        
        # 渲染
        rgb, mask = mujoco_left_arm.render("render_camera", 640, 480, with_mask=True)
        
        # 将Mask&RGB，并贴回原图显示
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        vis_combined = img.copy()
        vis_combined[mask_3ch > 0] = rgb[mask_3ch > 0]
        cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow("Mask", mask)
        cv2.imshow("Combined Visualization", vis_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 启动viewer
    print("\n[INFO] 启动MuJoCo Viewer...")
    mujoco_left_arm.spin()
