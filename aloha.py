#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALOHA风格主从示教控制器
4个机械臂：主1、主2（泄力状态），从1、从2（复刻主臂姿态）

使用方法：
    python aloha.py

CAN端口配置：
    - 主1: can0
    - 主2: can1  
    - 从1: can2
    - 从2: can3
"""

import time
import signal
import sys
import numpy as np
from typing import Optional, Dict, Any
from fucking_arx_mujoco.real.real_single_arm import RealSingleArm  # 使用官方底层接口

class AlohaTeaching:
    """
    ALOHA主从示教控制器
    
    主臂（Master）：泄力状态，人工拖动示教
    从臂（Follower）：实时复刻主臂姿态
    """
    
    def __init__(self, 
                 master1_can: str = 'can0',
                 master2_can: str = 'can1',
                 follower1_can: str = 'can2',
                 follower2_can: str = 'can3',
                 arm_type: int = 0,
                 control_freq: float = 50.0):
        """
        初始化ALOHA示教系统
        
        Args:
            master1_can: 主臂1的CAN端口
            master2_can: 主臂2的CAN端口
            follower1_can: 从臂1的CAN端口
            follower2_can: 从臂2的CAN端口
            arm_type: 机械臂类型 (0: X5liteaa0, 1: R5_master)
            control_freq: 控制频率 (Hz)
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.running = False
        
        print("=" * 60)
        print("ALOHA 主从示教系统初始化")
        print("=" * 60)
        
        # 初始化4个机械臂
        print(f"\n[1/4] 初始化主臂1 ({master1_can})...")
        self.master1 = self._create_arm(master1_can, arm_type, "Master1")
        
        print(f"\n[2/4] 初始化主臂2 ({master2_can})...")
        self.master2 = self._create_arm(master2_can, arm_type, "Master2")
        
        print(f"\n[3/4] 初始化从臂1 ({follower1_can})...")
        self.follower1 = self._create_arm(follower1_can, arm_type, "Follower1")
        
        print(f"\n[4/4] 初始化从臂2 ({follower2_can})...")
        self.follower2 = self._create_arm(follower2_can, arm_type, "Follower2")
        
        print("\n" + "=" * 60)
        print("所有机械臂初始化完成!")
        print("=" * 60)
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_arm(self, can_port: str, arm_type: int, name: str) -> RealSingleArm:
        """
        创建单个机械臂实例（使用 RealSingleArm）
        
        Args:
            can_port: CAN端口
            arm_type: 机械臂类型
            name: 机械臂名称（用于日志）
            
        Returns:
            SingleArm实例
        """
        try:
            arm = RealSingleArm(can_port=can_port, arm_type=arm_type, max_velocity=300, max_acceleration=800)
            print(f"  [{name}] 连接成功")
            return arm
        except Exception as e:
            print(f"  [{name}] 连接失败: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅退出"""
        print("\n\n收到退出信号，正在停止...")
        self.running = False
    
    def enable_master_gravity_compensation(self):
        """
        启用主臂重力补偿（泄力模式）
        主臂可以被人工自由拖动
        """
        print("\n启用主臂重力补偿模式...")
        try:
            self.master1.arm.gravity_compensation()
            print("  [Master1] 重力补偿已启用")
        except Exception as e:
            print(f"  [Master1] 启用失败: {e}")
        try:
            self.master2.arm.gravity_compensation()
            print("  [Master2] 重力补偿已启用")
        except Exception as e:
            print(f"  [Master2] 启用失败: {e}")

    def sync_followers_to_masters(self):
        """
        同步从臂到主臂当前位置
        在开始示教前调用，确保从臂与主臂位置一致
        """
        print("\n同步从臂到主臂位置...")
        try:
            master1_pose = self.master1.get_gripper_pose()
            master2_pose = self.master2.get_gripper_pose()
            self.follower1.set_gripper_pose(master1_pose)
            self.follower2.set_gripper_pose(master2_pose)
            print("  从臂已同步到主臂位置")
        except Exception as e:
            print(f"  同步失败: {e}")

    def go_home_all(self):
        """所有机械臂回零位"""
        print("\n所有机械臂回零位...")
        try:
            self.master1.go_home()
            self.master2.go_home()
            self.follower1.go_home()
            self.follower2.go_home()
            print("  回零位指令已发送")
        except Exception as e:
            print(f"  回零位失败: {e}")

    def start_teaching(self):
        """
        开始主从示教
        主臂泄力，从臂实时跟随（关节角度和夹爪宽度1:1复制）
        """
        print("\n" + "=" * 60)
        print("开始主从示教")
        print("=" * 60)
        print("操作说明:")
        print("  - 主臂已进入泄力模式，可以自由拖动")
        print("  - 从臂将实时复刻主臂的关节角度和夹爪宽度（1:1复制）")
        print("  - 按 Ctrl+C 停止示教")
        print("=" * 60 + "\n")
        
        # 启用主臂重力补偿
        self.enable_master_gravity_compensation()
        time.sleep(0.5)
        self.running = True
        loop_count = 0

        while self.running:
            t_start = time.time()
            try:
                # 读取主臂关节角度和夹爪宽度
                master1_joints = self.master1.get_joint_positions()
                master2_joints = self.master2.get_joint_positions()
                master1_gripper = self.master1.get_gripper_width(teacher=True)
                master2_gripper = self.master2.get_gripper_width(teacher=True)

                # 从臂复刻主臂（关节角度和夹爪宽度1:1复制）
                if master1_joints is not None and not np.isnan(master1_joints).any():
                    self.follower1.set_joint_positions(master1_joints)
                    self.follower1.set_gripper_width(master1_gripper)
                if master2_joints is not None and not np.isnan(master2_joints).any():
                    self.follower2.set_joint_positions(master2_joints)
                    self.follower2.set_gripper_width(master2_gripper)

                # 定期打印状态
                loop_count += 1
                if loop_count % int(self.control_freq) == 0:
                    print(f"[Master1] joints: {np.round(master1_joints, 3) if master1_joints is not None else 'None'}, gripper: {master1_gripper:.4f} | [Master2] joints: {np.round(master2_joints, 3) if master2_joints is not None else 'None'}, gripper: {master2_gripper:.4f}")
            except Exception as e:
                print(f"控制循环错误: {e}")
            elapsed = time.time() - t_start
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
        print("\n示教已停止")
        self._stop_all()

    def _stop_all(self):
        """停止所有机械臂"""
        print("\n停止所有机械臂...")
        try:
            # 可以选择让机械臂保持当前位置或回零位
            # self.go_home_all()
            pass
        except Exception as e:
            print(f"停止失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ALOHA主从示教系统')
    parser.add_argument('--master1', type=str, default='can0', 
                        help='主臂1的CAN端口')
    parser.add_argument('--master2', type=str, default='can2',
                        help='主臂2的CAN端口 ')
    parser.add_argument('--follower1', type=str, default='can3', 
                        help='从臂1的CAN端口')
    parser.add_argument('--follower2', type=str, default='can1', 
                        help='从臂2的CAN端口')
    
    args = parser.parse_args()
    
    # 创建示教系统
    aloha = AlohaTeaching(
        master1_can=args.master1,
        master2_can=args.master2,
        follower1_can=args.follower1,
        follower2_can=args.follower2
    )

    aloha.go_home_all()
    print("等待回零位完成...")
    
    # 开始示教
    aloha.start_teaching()


if __name__ == "__main__":
    main()
