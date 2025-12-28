import os
import cv2
import glob
import time
import numpy as np
from typing import List, Optional
from real.camera.aruco_utils import get_single_aruco
from real.camera.camera_utils import load_camera_intrinsics, load_eye_to_hand_matrix, T_optical_to_link
from real2sim import Real2Sim
from real.real_single_arm import RealSingleArm


def load_image_sequence(folder: str) -> List[str]:
    """加载文件夹中的图片序列并排序"""
    paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def detect_aruco_sequence(image_paths: List[str], K: np.ndarray, dist: np.ndarray,
                          marker_size: float, target_id: int) -> List[Optional[np.ndarray]]:
    """返回相机link坐标系下的位姿序列"""
    T_link_optical = T_optical_to_link()
    results = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            results.append(None)
            continue
        aruco = get_single_aruco(img, K, dist, marker_id=target_id, marker_size=marker_size)
        results.append(T_link_optical @ aruco.T_cam_marker if aruco else None)
    print(f"[INFO] ArUco检测: {sum(1 for r in results if r is not None)}/{len(results)} 帧有效")
    return results


def interpolate_missing_poses(T_list: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    """用最近有效位姿填充缺失帧"""
    result, last_valid = [], None
    for T in T_list:
        if T is not None:
            last_valid = T.copy()
        result.append(last_valid.copy() if last_valid is not None else np.eye(4))
    # 回填开头
    for i, T in enumerate(T_list):
        if T is not None:
            for j in range(i):
                result[j] = T.copy()
            break
    return result


if __name__ == "__main__":
    # 参数
    ARUCO_FOLDER = "assert/aruco_seqs"
    INTRINSICS_PATH = "real/camera/camera_intrinsics_d435i.json"
    EYE_TO_HAND_PATH = "real/camera/eye_to_hand_result_left_latest.json"
    XML_PATH = "SDK/R5a/meshes/R5a_R5master.xml"
    MARKER_SIZE, TARGET_ID = 0.05, 0
    
    # 真实机械臂参数
    USE_REAL_ARM = True
    CAN_PORT = "can1"
    MAX_VEL, MAX_ACC = 100, 300
    DEFAULT_GRIPPER_WIDTH = 30.0
    
    # 加载参数
    _, K, dist, v_fov = load_camera_intrinsics(INTRINSICS_PATH)
    T_flange_init_camlink = load_eye_to_hand_matrix(EYE_TO_HAND_PATH)
    
    # 加载图片
    image_paths = load_image_sequence(ARUCO_FOLDER)
    if not image_paths:
        print(f"[ERROR] 无图片: {ARUCO_FOLDER}")
        exit(1)
    print(f"[INFO] 加载 {len(image_paths)} 张图片")
    
    # 检测ArUco（相机link坐标系）
    T_camlink_ee_list = interpolate_missing_poses(
        detect_aruco_sequence(image_paths, K, dist, MARKER_SIZE, TARGET_ID)
    )
    
    # 转换到flange_init坐标系
    T_flange_init_ee_list = [T_flange_init_camlink @ T for T in T_camlink_ee_list]
    
    # 创建仿真渲染器
    sample = cv2.imread(image_paths[0])
    r2s = Real2Sim(
        xml_path=XML_PATH,
        T_flange_init_camlink=T_flange_init_camlink,
        width=sample.shape[1],
        height=sample.shape[0],
        fov=v_fov,
        verbose=False
    )
    
    # 批量渲染
    print("[INFO] 仿真渲染中...")
    results = r2s.render_batch(T_camlink_ee_list, show_progress=False)
    print(f"[INFO] 渲染完成，IK失败: {sum(1 for r in results if not r.ik_success)}/{len(results)}")
    
    # 初始化真实机械臂
    real_arm = None
    if USE_REAL_ARM:
        print(f"[INFO] 连接机械臂: {CAN_PORT}")
        real_arm = RealSingleArm(can_port=CAN_PORT, max_velocity=MAX_VEL, max_acceleration=MAX_ACC)
        time.sleep(1)
    
    # 可视化 + 执行
    print("[INFO] 操作: q=退出, 空格=暂停, a/d=前后帧, r=执行当前帧, h=回零")
    paused, idx = True, 0
    
    while idx < len(results):
        img = cv2.imread(image_paths[idx])
        res = results[idx]
        T_flange_init_ee = T_flange_init_ee_list[idx]
        
        # 1. 显示仿真结果
        rgb_bgr = cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR)
        overlay = img.copy()
        overlay[res.mask > 0] = rgb_bgr[res.mask > 0]
        
        pos = T_flange_init_ee[:3, 3]
        cv2.putText(overlay, f"{idx}/{len(results)-1} IK:{'OK' if res.ik_success else 'FAIL'}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(overlay, f"{'PAUSED' if paused else 'RUNNING'} | Real:{'ON' if USE_REAL_ARM else 'OFF'}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Real2Sim", overlay)
        
        # 2. 按键处理
        key = cv2.waitKey(0 if paused else 100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a') and paused:
            idx = max(0, idx - 1)
            continue
        elif key == ord('d') and paused:
            idx = min(len(results) - 1, idx + 1)
            continue
        elif key == ord('r'):
            # 执行当前帧
            if real_arm and res.ik_success:
                print(f"[INFO] 执行帧 {idx}")
                real_arm.move_to(T_flange_init_ee, DEFAULT_GRIPPER_WIDTH, is_gripper_pose=True)
            continue
        elif key == ord('h'):
            if real_arm:
                print("[INFO] 回零")
                real_arm.go_home()
            continue
        
        # 3. 非暂停：先显示，再执行真实机械臂
        if not paused:
            if real_arm and res.ik_success:
                real_arm.move_to(T_flange_init_ee, DEFAULT_GRIPPER_WIDTH, is_gripper_pose=True)
            idx += 1
    
    cv2.destroyAllWindows()
    
    if real_arm:
        print("[INFO] 回零")
        real_arm.go_home()
        time.sleep(2)
    
    r2s.spin()
