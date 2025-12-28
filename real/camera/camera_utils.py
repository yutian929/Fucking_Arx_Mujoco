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
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 畸变系数（取前5个，转为numpy array）
    disto = cam_data.get("disto", [0.0] * 5)
    dist = np.array(disto[:5], dtype=np.float64)
    
    return cam_data, K, dist


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