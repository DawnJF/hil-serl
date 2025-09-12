import json
import random
import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def image_normalization(img):
    # 如果图像是 HWC 格式，转换为 CHW 格式
    if img.shape[-1] == 3:  # 最后一个维度是通道数
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    normalized_img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    standardized_img = (normalized_img - mean) / std
    return standardized_img


def jax_process_images(observation):
    """
    Process images shape (1, H, W, C) to (C, H, W)

    Args:
        image (np.array): input image array, shape (1, H, W, C)

    Returns:
        np.array: output image array, shape (C, H, W)
    """
    if len(observation.shape) > 3:
        observation = observation.squeeze(0)
    if observation.shape[2] == 3:
        observation = observation.transpose(2, 0, 1)
    return observation


def jax_process_trajectory(data_path):
    """
    For data of multiple trajectories collected in a single session

    data keys: ['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones', 'infos']
    observations keys: ['rgb', 'state', 'wrist']

    Args:
        data_path (str): path to the pickle data file

    Returns:
        transitions (list): transitions data
    """
    transitions = []
    df = pd.read_pickle(data_path)
    for i in tqdm(range(len(df)), desc="Processing trajectories"):
        observation = df[i]["observations"]["rgb"]
        next_observation = df[i]["next_observations"]["rgb"]
        observation = jax_process_images(observation)
        next_observation = jax_process_images(next_observation)
        reward = df[i]["rewards"]
        transition = {
            "observations": observation,
            "next_observations": next_observation,
            "reward": reward,
        }
        transitions.append(transition)
    return transitions


def jax_load_and_convert_data(data_path, folder_list=None):
    """
    data_path: str, eg. "/media/robot/30F73268F87D0FEF/Jax_Hil_Serl_Dataset/2025-09-01/"
    folder_list: list of str, eg. ["usb_pickup_insertion_5_14-42-28.pkl"]
    """
    print("开始加载数据...")
    transitions = []
    folder_list = os.listdir(data_path) if folder_list is None else folder_list
    for folder in folder_list:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在: {folder_path}")
            continue
        processed_transitions = jax_process_trajectory(
            data_path=folder_path,
        )
        transitions.extend(processed_transitions)
    return transitions


def load_pkl_to_reward_model(test_size=0.3, random_state=42):
    folder_list = ["merged_data.pkl"]
    data_path = "/home/facelesswei/code/hil-serl/classifier_data/09_11_4data"
    transitions = jax_load_and_convert_data(data_path, folder_list)
    images_list = []
    class_list = []
    for transition in transitions:
        image = transition["observations"]
        images_list.append([image])
        class_list.append(transition["reward"])

    # split 70% for training
    train_images, test_images, train_labels, test_labels = train_test_split(
        images_list,
        class_list,
        test_size=test_size,
        random_state=random_state,
        stratify=class_list,  # 保持各类别比例
    )
    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":

    load_pkl_to_reward_model()
