#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco码检测与可视化工具模块
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


# =========================
# 默认参数
# =========================
DEFAULT_MARKER_SIZE = 0.05  # 50mm -> meters
DEFAULT_MARKER_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class ArucoResult:
    """ArUco检测结果"""
    marker_id: int
    corners: np.ndarray          # (4, 2) 图像角点
    center: np.ndarray           # (2,) 中心点像素坐标
    rvec: np.ndarray             # (3,) 旋转向量
    tvec: np.ndarray             # (3,) 平移向量 (米)
    T_cam_marker: np.ndarray     # (4, 4) marker->camera 变换矩阵


def estimate_pose_from_corners(
    corners_4x2: np.ndarray,
    marker_size: float,
    K: np.ndarray,
    dist: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从角点估计ArUco码位姿
    
    Args:
        corners_4x2: (4,2) 图像角点 (左上, 右上, 右下, 左下)
        marker_size: marker边长 (米)
        K: 相机内参矩阵 (3,3)
        dist: 畸变系数 (5,)
    
    Returns:
        rvec: (3,) 旋转向量，失败返回None
        tvec: (3,) 平移向量，失败返回None
    """
    L = float(marker_size)
    
    # ArUco角点顺序：左上, 右上, 右下, 左下
    objp = np.array([
        [-L/2,  L/2, 0.0],
        [ L/2,  L/2, 0.0],
        [ L/2, -L/2, 0.0],
        [-L/2, -L/2, 0.0],
    ], dtype=np.float64)
    
    imgp = np.asarray(corners_4x2, dtype=np.float64).reshape(4, 2)
    
    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
    
    if not ok:
        return None, None
    return rvec.reshape(3), tvec.reshape(3)


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    将rvec/tvec转换为4x4变换矩阵 T_cam_marker
    """
    from scipy.spatial.transform import Rotation as R
    
    T = np.eye(4, dtype=np.float64)
    Rm = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def detect_aruco(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_size: float = DEFAULT_MARKER_SIZE,
    marker_dict: int = DEFAULT_MARKER_DICT,
    target_id: Optional[int] = None
) -> List[ArucoResult]:
    """
    检测图像中的ArUco码并估计位姿
    
    Args:
        img: BGR图像
        K: 相机内参矩阵 (3,3)
        dist: 畸变系数 (5,)
        marker_size: marker边长 (米)
        marker_dict: ArUco字典类型
        target_id: 指定只返回某个ID的marker，None表示返回所有
    
    Returns:
        ArucoResult列表
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    dictionary = cv2.aruco.getPredefinedDictionary(marker_dict)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    
    corners, ids, _ = detector.detectMarkers(gray)
    
    results = []
    
    if ids is None or len(ids) == 0:
        return results
    
    for i, marker_id in enumerate(ids.flatten()):
        if target_id is not None and marker_id != target_id:
            continue
        
        corner = corners[i].reshape(4, 2)
        center = corner.mean(axis=0)
        
        rvec, tvec = estimate_pose_from_corners(corner, marker_size, K, dist)
        
        if rvec is None:
            continue
        
        T_cam_marker = rvec_tvec_to_T(rvec, tvec)
        
        results.append(ArucoResult(
            marker_id=int(marker_id),
            corners=corner,
            center=center,
            rvec=rvec,
            tvec=tvec,
            T_cam_marker=T_cam_marker
        ))
    
    return results


def draw_aruco(
    img: np.ndarray,
    results: List[ArucoResult],
    K: np.ndarray,
    dist: np.ndarray,
    axis_length: float = 0.04,
    draw_corners: bool = True,
    draw_axes: bool = True,
    draw_info: bool = True
) -> np.ndarray:
    """
    在图像上可视化ArUco检测结果
    
    Args:
        img: BGR图像
        results: detect_aruco返回的结果列表
        K: 相机内参矩阵
        dist: 畸变系数
        axis_length: 坐标轴长度 (米)
        draw_corners: 是否绘制角点
        draw_axes: 是否绘制坐标轴
        draw_info: 是否绘制ID和位置信息
    
    Returns:
        标注后的图像
    """
    vis = img.copy()
    
    for res in results:
        # 绘制角点框
        if draw_corners:
            pts = res.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
            # 标记四个角点
            for j, pt in enumerate(res.corners):
                color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][j]
                cv2.circle(vis, tuple(pt.astype(int)), 5, color, -1)
        
        # 绘制坐标轴
        if draw_axes:
            cv2.drawFrameAxes(vis, K, dist, res.rvec, res.tvec, axis_length)
        
        # 绘制ID和位置信息
        if draw_info:
            cx, cy = res.center.astype(int)
            cv2.circle(vis, (cx, cy), 5, (255, 0, 255), -1)
            
            # ID
            cv2.putText(vis, f"ID:{res.marker_id}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 位置信息
            tx, ty, tz = res.tvec
            info_text = f"[{tx:.3f}, {ty:.3f}, {tz:.3f}]m"
            cv2.putText(vis, info_text, (cx + 10, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis


def get_single_aruco(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_id: int = 0,
    marker_size: float = DEFAULT_MARKER_SIZE,
    marker_dict: int = DEFAULT_MARKER_DICT
) -> Optional[ArucoResult]:
    """
    便捷函数：检测并返回指定ID的单个ArUco码
    
    Args:
        img: BGR图像
        K: 相机内参矩阵
        dist: 畸变系数
        marker_id: 目标marker ID
        marker_size: marker边长 (米)
        marker_dict: ArUco字典类型
    
    Returns:
        ArucoResult或None
    """
    results = detect_aruco(img, K, dist, marker_size, marker_dict, target_id=marker_id)
    return results[0] if results else None


# =========================
# 测试脚本
# =========================
if __name__ == "__main__":
    import os
    
    # =========================
    # 直接设定参数
    # =========================
    IMAGE_PATH = "/home/yutian/Hand2Gripper_phantom/submodules/Fucking_Arx_Mujoco/assert/calib_screenshot_raw_20251228_091850.png"  # 输入图片路径
    MARKER_SIZE = 0.05             # Marker边长(米)
    TARGET_MARKER_ID = 0           # 指定检测的marker ID，None表示检测所有
    OUTPUT_PATH = None             # 输出图片路径，None表示直接显示
    
    # 相机内参（使用实际标定值，不要用图片中心近似）
    FX = 606.0810546875
    FY = 605.1178588867188
    CX = 327.5788879394531
    CY = 245.88775634765625
    
    # =========================
    # 构造相机内参
    # =========================
    K = np.array([
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    
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
    print(f"[INFO] 使用相机内参: fx={FX}, fy={FY}, cx={CX}, cy={CY}")
    
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
    
    # 保存或显示
    if OUTPUT_PATH:
        cv2.imwrite(OUTPUT_PATH, vis)
        print(f"[OK] 保存结果到: {OUTPUT_PATH}")
    else:
        cv2.imshow("ArUco Detection", vis)
        print("[INFO] 按任意键退出")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
