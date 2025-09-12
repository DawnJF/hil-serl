import json
import random
import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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


def load_pkl(data_path, key_mapping):
    """
    data keys: ['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones', 'infos']
    observations keys: ['rgb', 'state', 'wrist']

    Args:
        key_mapping (dict): mapping from old keys to new keys
    Returns:
        transitions (list): transitions data
    """
    transitions = []
    df = pd.read_pickle(data_path)
    for i in tqdm(range(len(df)), desc="Processing pkl data"):
        transition = {}

        # Iterate over key_mapping to re-map the keys based on user-defined mapping
        for old_key, new_key in key_mapping.items():
            # Check if the key is a top-level key or nested under 'observations' (or other nested keys)
            keys = old_key.split(":")  # This assumes the key is like "observations:rgb"

            if len(keys) == 1:
                key = keys[0]
                assert key in df[i]

                transition[new_key] = df[i][key]
            elif len(keys) == 2:
                top_key, sub_key = keys
                assert top_key in df[i] and sub_key in df[i][top_key]

                transition[new_key] = df[i][top_key][sub_key]

        transitions.append(transition)
    return transitions


def test_load_pkl():
    mapping = {
        "observations:rgb": "image1",
        "observations:wrist": "image2",
        "observations:state": "state",
        "actions": "actions",
    }
    load_pkl("/Users/majianfei/Downloads/usb_pickup_insertion_30_11-50-21.pkl", mapping)


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


def load_pkl_to_reward_model(path, test_size=0.3, random_state=42):
    folder_list = [os.path.basename(path)]
    data_path = os.path.dirname(path)
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

    test_load_pkl()
