import os
import cv2
import glob
import numpy as np
from typing import List, Optional
from real.camera.aruco_utils import detect_aruco, draw_aruco, get_single_aruco
from real.camera.camera_utils import load_camera_intrinsics, load_eye_to_hand_matrix, T_optical_to_link
from real2sim import Real2Sim


def load_image_sequence(folder: str) -> List[str]:
    """加载文件夹中的图片序列并排序"""
    paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def detect_aruco_sequence(image_paths: List[str], K: np.ndarray, dist: np.ndarray,
                          marker_size: float, target_id: int) -> List[Optional[np.ndarray]]:
    """从图片序列中检测ArUco码，返回相机link坐标系下的位姿序列"""
    T_link_optical = T_optical_to_link()
    results = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            results.append(None)
            continue
        
        aruco = get_single_aruco(img, K, dist, marker_id=target_id, marker_size=marker_size)
        if aruco is None:
            results.append(None)
        else:
            results.append(T_link_optical @ aruco.T_cam_marker)
    
    valid = sum(1 for r in results if r is not None)
    print(f"[INFO] ArUco检测: {valid}/{len(results)} 帧有效")
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
    
    # 加载参数
    _, K, dist, v_fov = load_camera_intrinsics(INTRINSICS_PATH)
    T_flange_init_camlink = load_eye_to_hand_matrix(EYE_TO_HAND_PATH)
    
    # 加载图片
    image_paths = load_image_sequence(ARUCO_FOLDER)
    if not image_paths:
        print(f"[ERROR] 无图片: {ARUCO_FOLDER}")
        exit(1)
    print(f"[INFO] 加载 {len(image_paths)} 张图片")
    
    # 检测ArUco
    T_camlink_list = detect_aruco_sequence(image_paths, K, dist, MARKER_SIZE, TARGET_ID)
    if all(T is None for T in T_camlink_list):
        print("[ERROR] 未检测到ArUco")
        exit(1)
    
    T_camlink_ee_list = interpolate_missing_poses(T_camlink_list)
    
    # 创建渲染器
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
    print("[INFO] 渲染中...")
    results = r2s.render_batch(T_camlink_ee_list, show_progress=False)
    print(f"[INFO] 渲染完成，IK失败: {sum(1 for r in results if not r.ik_success)}/{len(results)}")
    
    # 可视化
    print("[INFO] 显示结果 (q:退出, 空格:暂停, a/d:前后帧)")
    paused, idx = False, 0
    
    while idx < len(results):
        img = cv2.imread(image_paths[idx])
        res = results[idx]
        
        # 叠加渲染结果
        rgb_bgr = cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR)
        overlay = img.copy()
        overlay[res.mask > 0] = rgb_bgr[res.mask > 0]
        
        # 状态
        status = "OK" if res.ik_success else "FAIL"
        cv2.putText(overlay, f"{idx}/{len(results)-1} IK:{status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Real2Sim", overlay)
        
        key = cv2.waitKey(0 if paused else 500) & 0xFF
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
        
        if not paused:
            idx += 1
    
    cv2.destroyAllWindows()
    r2s.spin()
