#!/usr/bin/env python3
"""
分析训练数据中动作值的范围
"""
import sys
import os
import numpy as np

sys.path.append(os.getcwd())
from reward_model.pkl_utils import load_pkl


def analyze_actions():
    # 加载数据
    mapping = {
        "observations:rgb": "image1",
        "observations:wrist": "image2",
        "observations:state": "state",
        "actions": "actions",
    }

    data_files = ["/Users/majianfei/Downloads/usb_pickup_insertion_5_11-05-02.pkl"]

    all_actions = []

    for file_pattern in data_files:
        if not os.path.exists(file_pattern):
            print(f"文件不存在：{file_pattern}")
            continue

        try:
            file_data = load_pkl(file_pattern, mapping)

            for item in file_data:
                action = item["actions"]
                if len(action) >= 3:  # 只分析前3个连续动作维度
                    all_actions.append(action[:3])

            print(f"加载 {file_pattern}: {len(file_data)} 个样本")
        except Exception as e:
            print(f"加载失败 {file_pattern}: {e}")
            continue

    if not all_actions:
        print("没有找到有效的动作数据")
        return

    all_actions = np.array(all_actions)

    print(f"\n动作数据统计:")
    print(f"总样本数: {len(all_actions)}")
    print(f"动作维度: {all_actions.shape[1]}")

    for i in range(all_actions.shape[1]):
        action_dim = all_actions[:, i]
        print(f"\n维度 {i}:")
        print(f"  最小值: {action_dim.min():.6f}")
        print(f"  最大值: {action_dim.max():.6f}")
        print(f"  均值: {action_dim.mean():.6f}")
        print(f"  标准差: {action_dim.std():.6f}")
        print(f"  中位数: {np.median(action_dim):.6f}")
        print(
            f"  超出[-1,1]范围的样本数: {np.sum((action_dim < -1) | (action_dim > 1))}"
        )
        print(
            f"  超出比例: {np.sum((action_dim < -1) | (action_dim > 1)) / len(action_dim) * 100:.2f}%"
        )


if __name__ == "__main__":
    analyze_actions()
