#!/usr/bin/env python3
"""
将take5数据转换为HIL-SERL格式的演示数据
"""

import json
import pickle as pkl
import numpy as np
import os
from PIL import Image
from pathlib import Path


def process_trajectory(transitions, data_path, image_base_path, success_index):
    # 加载JSON数据
    with open(os.path.join(data_path, "data.json"), "r") as f:
        data = json.load(f)

    print(f"加载了 {len(data)} 个数据点")
    for i in range(len(data) - 1):  # 最后一个数据点没有next_observation
        current_frame = data[i]
        next_frame = data[i + 1]

        # 加载当前和下一帧的图像
        current_image_path = os.path.join(
            image_base_path, current_frame["front_head"].lstrip("/")
        )
        next_image_path = os.path.join(
            image_base_path, next_frame["front_head"].lstrip("/")
        )

        # 检查图像文件是否存在
        if not os.path.exists(current_image_path):
            print(f"警告: 图像文件不存在: {current_image_path}")
            continue
        if not os.path.exists(next_image_path):
            print(f"警告: 图像文件不存在: {next_image_path}")
            continue

        # 加载图像并缩放到128x128 (PIL -> numpy)
        current_image_pil = Image.open(current_image_path)
        next_image_pil = Image.open(next_image_path)

        # 缩放图像到128x128
        current_image_pil = current_image_pil.resize(
            (128, 128), Image.Resampling.LANCZOS
        )
        next_image_pil = next_image_pil.resize((128, 128), Image.Resampling.LANCZOS)

        # 转换为numpy数组
        current_image = np.array(current_image_pil)
        next_image = np.array(next_image_pil)

        # 确保图像是HWC格式 (Height, Width, Channels)
        if len(current_image.shape) == 3:
            if current_image.shape[0] == 3:  # 如果是CHW，转换为HWC
                current_image = current_image.transpose(1, 2, 0)
        if len(next_image.shape) == 3:
            if next_image.shape[0] == 3:  # 如果是CHW，转换为HWC
                next_image = next_image.transpose(1, 2, 0)

        # 构建观测数据
        observations = {"image": current_image}

        next_observations = {"image": next_image}

        # 动作数据 (使用joint_angles)
        actions = np.array(current_frame["joint_angles"], dtype=np.float32)

        if i >= success_index:
            reward = 1.0  # 奖励 (由于是演示数据，给予正奖励)
        else:
            reward = 0.0

        # episode结束标志 (最后一个转换标记为结束)
        done = i == len(data) - 2

        # 构建转换
        transition = {
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "rewards": reward,
            "dones": done,
            "masks": 1.0 - done,  # masks = 1.0 - done
        }

        transitions.append(transition)


def load_and_convert_data(
    data_path,
    folder_list,
    output_path="/Users/majianfei/Projects/Github/ML/hil-serl/dataset/success_demo.pkl",
):
    """
    转换take5数据格式为HIL-SERL格式

    Args:
        data_json_path: JSON数据文件路径
        image_base_path: 图像文件基础路径
        output_path: 输出pkl文件路径
    """
    print("开始加载数据...")

    transitions = []

    for folder, success_index in folder_list:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在: {folder_path}")
            continue

        # 处理每个文件夹中的数据
        data_json_path = os.path.join(folder_path, "data.json")
        if not os.path.exists(data_json_path):
            print(f"警告: JSON文件不存在: {data_json_path}")
            continue

        process_trajectory(
            transitions,
            data_path=folder_path,
            image_base_path=folder_path,
            success_index=success_index,
        )
        print(f"已处理: {len(transitions)} 个转换来自文件夹 {folder}")

    # 转换为轨迹格式

    print(f"总共创建了 {len(transitions)} 个转换")

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 保存为pickle文件
    with open(output_path, "wb") as f:
        pkl.dump(transitions, f)

    print(f"数据已保存到: {output_path}")

    # 打印数据统计信息
    print("\n数据统计信息:")
    print(f"转换数量: {len(transitions)}")
    print(f"动作维度: {transitions[0]['actions'].shape}")
    print(f"图像形状: {transitions[0]['observations']['image'].shape}")
    print()
    print(
        f"动作范围: [{transitions[0]['actions'].min():.4f}, {transitions[0]['actions'].max():.4f}]"
    )

    return transitions


def validate_data(pkl_path):
    """验证转换后的数据"""
    print(f"\n验证数据: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    print(f"加载了 {len(data)} 个转换")

    # 检查第一个转换
    first_transition = data[0]
    print(f"第一个转换的键: {list(first_transition.keys())}")
    print(f"观测键: {list(first_transition['observations'].keys())}")
    print(f"图像形状: {first_transition['observations']['image'].shape}")
    print(f"动作形状: {first_transition['actions'].shape}")
    print(f"奖励: {first_transition['rewards']}")
    print(f"done: {first_transition['dones']}")
    print(f"mask: {first_transition['masks']}")

    # 检查数据类型
    print(f"图像数据类型: {first_transition['observations']['image'].dtype}")
    print(f"动作数据类型: {first_transition['actions'].dtype}")

    # 检查动作统计
    all_actions = np.array([t["actions"] for t in data])
    print(f"所有动作形状: {all_actions.shape}")
    print(f"动作统计:")
    print(f"  最小值: {all_actions.min(axis=0)}")
    print(f"  最大值: {all_actions.max(axis=0)}")
    print(f"  均值: {all_actions.mean(axis=0)}")
    print(f"  标准差: {all_actions.std(axis=0)}")


if __name__ == "__main__":

    # folder_list = [
    #     ("take5", 122),
    #     ("take17", 110),
    #     ("take11", 123),
    #     ("take31", 97),
    #     ("take67", 138),
    #     ("take107", 96),
    #     ("take109", 108),
    #     ("take132", 112),
    #     ("take174", 106),
    #     ("take197", 106),
    #     ("take204", 81),
    #     ("take216", 96),
    #     ("take246", 88),
    #     ("take305", 108),
    #     ("take341", 97),
    #     ("take389", 148),
    #     ("take485", 195),
    # ]
    folder_list_v2 = [
        ("take5", 122),
        ("take17", 110),
        ("take11", 123),
        ("take31", 97),
        ("take67", 138),
        ("take107", 96),
        ("take109", 108),
        ("take132", 112),
        ("take174", 106),
        ("take197", 106),
        ("take204", 81),
        ("take216", 96),
        ("take246", 88),
        ("take305", 108),
        ("take341", 97),
        ("take389", 148),
        ("take485", 195),
        ############
        ("take503", 76),
        ("take509", 97),
        ("take510", 97),
        ("take511", 129),
        ("take515", 81),
        ("take516", 71),
        ("take535", 85),
        ("take540", 86),
        ("take545", 121),
        ("take552", 94),
    ]
    # 转换数据
    transitions = load_and_convert_data(
        data_path="/liujinxin/dataset/bimanual/0707_tidy_tools_filtered/0707_new_1",
        folder_list=folder_list_v2,
        output_path=f"/liujinxin/mjf/hil-serl/dataset/success_demo_{len(folder_list_v2)}.pkl",
    )

    print("\n转换完成!")
