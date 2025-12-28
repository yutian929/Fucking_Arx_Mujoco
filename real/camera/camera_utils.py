#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机参数加载工具模块
"""

import json
import numpy as np
from typing import Tuple, Optional


def load_camera_intrinsics(
    json_path: str,
    camera: str = "left"
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    从JSON文件加载相机内参
    
    Args:
        json_path: JSON文件路径
        camera: 相机选择 ("left" 或 "right")
    
    Returns:
        raw_dict: 原始字典数据
        K: 相机内参矩阵 (3,3) numpy array
        dist: 畸变系数 (5,) numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cam_data = data.get(camera, data)
    
    # 构造内参矩阵
    fx = cam_data["fx"]
    fy = cam_data["fy"]
    cx = cam_data["cx"]
    cy = cam_data["cy"]
    v_fov = cam_data.get("v_fov", {})
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 畸变系数（取前5个，转为numpy array）
    disto = cam_data.get("disto", [0.0] * 5)
    dist = np.array(disto[:5], dtype=np.float64)
    
    return cam_data, K, dist, v_fov


def get_camera_intrinsics_from_dict(
    cam_data: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从字典构造相机内参矩阵和畸变系数
    
    Args:
        cam_data: 包含fx, fy, cx, cy, disto的字典
    
    Returns:
        K: 相机内参矩阵 (3,3) numpy array
        dist: 畸变系数 (5,) numpy array
    """
    fx = cam_data["fx"]
    fy = cam_data["fy"]
    cx = cam_data["cx"]
    cy = cam_data["cy"]
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    disto = cam_data.get("disto", [0.0] * 5)
    dist = np.array(disto[:5], dtype=np.float64)
    
    return K, dist


def load_eye_to_hand_matrix(json_path: str) -> np.ndarray:
    """
    加载手眼标定矩阵（相机link在机械臂基座坐标系下的位姿）
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        T_base_camlink: (4,4) numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["Mat_base_T_camera_link"], dtype=np.float64)


def T_optical_to_link() -> np.ndarray:
    """
    从optical坐标系到link坐标系的变换矩阵
    
    Optical (OpenCV): X-Right, Y-Down, Z-Forward
    Link (ROS):       X-Forward, Y-Left, Z-Up
    
    Returns:
        T_link_optical: (4,4) 变换矩阵，使得 P_link = T @ P_optical
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [ 0,  0,  1],  # link X = optical Z
        [-1,  0,  0],  # link Y = -optical X
        [ 0, -1,  0],  # link Z = -optical Y
    ], dtype=np.float64)
    return T


def T_link_to_optical() -> np.ndarray:
    """
    从link坐标系到optical坐标系的变换矩阵
    
    Returns:
        T_optical_link: (4,4) 变换矩阵
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [ 0, -1,  0],  # optical X = -link Y
        [ 0,  0, -1],  # optical Y = -link Z
        [ 1,  0,  0],  # optical Z = link X
    ], dtype=np.float64)
    return T
