import os
import cv2
import numpy as np
from typing import List, Optional
from real.camera.aruco_utils import detect_aruco, draw_aruco, ArucoResult
from real.camera.camera_utils import load_camera_intrinsics


if __name__ == "__main__":
    # =========================
    # 直接设定参数
    # =========================
    IMAGE_PATH = "assert/calib_screenshot_raw_20251228_091850.png"  # 输入图片路径
    CAMERA_INTRINSICS_PATH = "real/camera/camera_intrinsics_d435i.json"  # 相机内参JSON文件路径
    MARKER_SIZE = 0.05             # Marker边长(米)
    TARGET_MARKER_ID = 0           # 指定检测的marker ID，None表示检测所有
    
    _, K, dist = load_camera_intrinsics(CAMERA_INTRINSICS_PATH)
    
    # =========================
    # 读取图片
    # =========================
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] 图片不存在: {IMAGE_PATH}")
        exit(1)
    
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] 无法读取图片: {IMAGE_PATH}")
        exit(1)
    
    print(f"[INFO] 读取图片: {IMAGE_PATH}, 尺寸: {img.shape}")
    print(f"[INFO] 使用相机内参: \n{K}\n畸变系数: {dist}")
    
    # 注意：不要覆盖K的cx/cy，应使用实际标定的内参值
    
    # =========================
    # 检测ArUco
    # =========================
    results = detect_aruco(img, K, dist, MARKER_SIZE, target_id=TARGET_MARKER_ID)
    
    if not results:
        print("[WARN] 未检测到ArUco码")
    else:
        print(f"[INFO] 检测到 {len(results)} 个ArUco码:")
        for res in results:
            print(f"  - ID={res.marker_id}")
            print(f"    center(px): [{res.center[0]:.1f}, {res.center[1]:.1f}]")
            print(f"    tvec(m):    [{res.tvec[0]:.4f}, {res.tvec[1]:.4f}, {res.tvec[2]:.4f}]")
            print(f"    rvec(rad):  [{res.rvec[0]:.4f}, {res.rvec[1]:.4f}, {res.rvec[2]:.4f}]")
            print(f"    T_cam_marker:\n{res.T_cam_marker}")
    
    # =========================
    # 可视化
    # =========================
    vis = draw_aruco(img, results, K, dist)
    
    cv2.imshow("ArUco Detection", vis)
    print("[INFO] 按任意键退出")
    cv2.waitKey(0)
    cv2.destroyAllWindows()