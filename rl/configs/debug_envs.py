import shutil
import time
import sys
import os
import pickle as pkl
from PIL import Image
import numpy as np
import glob

sys.path.append(os.getcwd())
from rl import mappings
from utils.tools import print_dict_structure


def debug_config():
    env = mappings.CONFIG_MAPPING[
        "plug_into_socket_with_power_cord"
    ]().get_environment()

    print("==== observation_space ====")
    print_dict_structure(env.observation_space)

    print("==== action_space ====")
    print(env.action_space)

    print("==== reset ====")

    obs, info = env.reset()
    print(obs.keys())
    print_dict_structure(obs)
    # print(f"obs['images']['wrist'].shape: {obs['images']['wrist'].shape}")
    # print(f"obs['state']['tcp_pose'].shape: {obs['state']['tcp_pose'].shape}")

    print("==== step ====")
    for _ in range(10000):
        start_time = time.perf_counter()
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        print(f"action: {action}")
        print(f"obs: state: {obs['state']}")
        dt_s = time.perf_counter() - start_time
        print(f"dt_s: {dt_s}")


def check_envs_obs():
    env = mappings.CONFIG_MAPPING["plug_into_socket_with_power_cord"]().get_environment(
        debug=True
    )

    print("==== observation_space ====")
    print_dict_structure(env.observation_space)

    print("==== action_space ====")
    print(env.action_space)

    print("==== reset ====")

    obs, info = env.reset()
    print(obs.keys())
    print_dict_structure(obs)

    folder = "outputs/debug_check_obs_images"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

    action = env.action_space.sample()
    for i in range(2000):
        start_time = time.perf_counter()

        obs, reward, done, truncated, info = env.step(np.zeros_like(action))

        if "intervene_action" not in info:
            continue

        # image_w = Image.fromarray(obs["wrist"].squeeze())
        # image_w.save(f"{folder}/image_w{i}.png")
        image_r = Image.fromarray(obs["rgb"].squeeze())
        image_r.save(f"{folder}/image_r{i}.png")
        image_s = Image.fromarray(obs["scene"].squeeze())
        image_s.save(f"{folder}/image_s{i}.png")


def check_dataset_obs():
    path = "datasets/trajectories/2025-11-07/traj_11-31-48_6.pkl"

    folder = "outputs/debug_check_obs_images"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

    with open(path, "rb") as f:
        transitions = pkl.load(f)
        print(transitions[0].keys())

        for i, t in enumerate(transitions):

            image_s = Image.fromarray(t["observations"]["scene"].squeeze())
            image_s.save(f"{folder}/image_s{i}.png")


def replay_dataset():
    path = "datasets/trajectories/2025-11-07/traj_11-31-48_6.pkl"

    env = mappings.CONFIG_MAPPING["open_switch"]().get_environment()

    env.reset()

    with open(path, "rb") as f:
        transitions = pkl.load(f)
        print(transitions[0].keys())

        for i in range(len(transitions)):
            print("==== step ", i, " ====")
            print(f"reward: {transitions[i]['rewards']}")
            print(f"done: {transitions[i]['dones']}")
            print(f"action: {transitions[i]['actions']}")
            print(f"masks: {transitions[i]['masks']}")

            env.step(transitions[i]["actions"])
            if transitions[i]["dones"]:
                print(f"Episode {i} finished.")
                break


def count_buffer_size():
    checkpoint_path = "outputs/mse/plug/"
    # checkpoint_path = "outputs/rlpd/debug1"
    buffer_size = 0
    for file in glob.glob(os.path.join(checkpoint_path, "buffer/*.pkl")):
        with open(file, "rb") as f:
            transitions = pkl.load(f)
            print(f"File: {file}, size: {len(transitions)}")
            buffer_size += len(transitions)
    print(f"Total buffer size: {buffer_size}")

    demo_buffer_size = 0
    for file in glob.glob(os.path.join(checkpoint_path, "demo_buffer/*.pkl")):
        with open(file, "rb") as f:
            transitions = pkl.load(f)
            print(f"File: {file}, size: {len(transitions)}")
            demo_buffer_size += len(transitions)
    print(f"Total demo buffer size: {demo_buffer_size}")


if __name__ == "__main__":
    # debug_config()
    # replay_dataset()
    # check_envs_obs()
    # check_dataset_obs()
    count_buffer_size()
