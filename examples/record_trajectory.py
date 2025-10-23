# 需要把环境中reset的time.sleep()注释一下   推理脚本不使用reward
# /home/facelesswei/code/hil-serl/examples/experiments/usb_pickup_insertion/config.py  MAX_EPISODE_LENGTH = 200改为10000000

import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from pynput import keyboard
import atexit

from experiments.mappings import CONFIG_MAPPING

# 状态标记
collecting = False  # 是否在采集轨迹
start_collect = False
stop_collect = False
delete_last = False  # 按 d 删除上一次轨迹


def on_press(key):
    global start_collect, stop_collect, delete_last
    try:
        if hasattr(key, "char"):
            if key.char == "a":
                start_collect = True
            elif key.char == "b":
                stop_collect = True
            elif key.char == "d":
                delete_last = True
    except AttributeError:
        pass


def setup_keyboard_listener():
    """设置键盘监听器"""
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    atexit.register(lambda: listener.stop())
    return listener


def save_trajectory(trajectory, save_dir, time_name, saved_count):
    """保存轨迹到文件"""
    file_name = os.path.join(save_dir, f"traj_{time_name}_{saved_count+1}.pkl")
    with open(file_name, "wb") as f:
        pkl.dump(trajectory, f)
    print(f"=====saved trajectory {saved_count+1} to {file_name}")
    return file_name


def handle_delete_trajectory(save_dir, time_name, saved_count, pbar):
    """处理删除上一条轨迹的逻辑"""
    if saved_count > 0:
        prev_file = os.path.join(save_dir, f"traj_{time_name}_{saved_count}.pkl")
        if os.path.exists(prev_file):
            os.remove(prev_file)
            print(f"Deleted previous trajectory: {prev_file}")
            saved_count -= 1
            pbar.update(-1)
    else:
        print("No previous trajectory to delete")
    return saved_count


def create_transition(obs, actions, next_obs, rew, done, info):
    """创建transition数据结构"""
    return copy.deepcopy(
        dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rew,
            masks=1.0 - done,
            dones=done,
            infos=info,
        )
    )


def collection_loop(env, save_dir, trajectories_needed):
    """主要的数据收集循环"""
    global collecting, start_collect, stop_collect, delete_last

    obs, info = env.reset()
    trajectory = []
    time_name = datetime.datetime.now().strftime("%H-%M-%S")
    start_idx = 1
    saved_count = start_idx - 1

    print(f"继续编号，从 {start_idx} 开始保存")
    pbar = tqdm(total=trajectories_needed, initial=saved_count)

    while saved_count < trajectories_needed:
        # 处理删除上一条轨迹
        if delete_last:
            saved_count = handle_delete_trajectory(
                save_dir, time_name, saved_count, pbar
            )
            delete_last = False

        # 获取动作和环境step
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        # 跳过无效动作
        if (
            not done
            and not rew
            and actions[0] == 0
            and actions[1] == 0
            and actions[2] == 0
            and actions[3] == 0
        ):
            continue

        # 创建transition
        transition = create_transition(obs, actions, next_obs, rew, done, info)
        obs = next_obs

        # 处理开始采集
        if start_collect:
            print("Start collecting new trajectory")
            if collecting:
                print("!!! Already collecting, resetting trajectory")
            trajectory = []
            start_collect = False

        # 处理停止采集并保存
        if stop_collect and collecting and len(trajectory) > 0:
            save_trajectory(trajectory, save_dir, time_name, saved_count)

            # 重置状态
            trajectory = []
            collecting = False
            stop_collect = False
            saved_count += 1
            pbar.update(1)

            # 重置环境
            obs, info = env.reset()
            print("====================收集了一条数据")

        if collecting:
            trajectory.append(transition)

        # 环境自然结束时重置
        if done or truncated:
            obs, info = env.reset()
            print("环境自然结束，已重置")

    print(f"Finished! Saved {saved_count} trajectories in {save_dir}")
    return saved_count


def main():
    dataset_folder = "datasets/trajectories"
    exp_name = "usb_pickup_insertion"
    trajectories_needed = 500

    # 设置键盘监听器
    listener = setup_keyboard_listener()

    # 初始化环境
    assert exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment()

    # 创建保存目录
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(dataset_folder, date_str)
    os.makedirs(save_dir, exist_ok=True)

    # 开始数据收集
    collection_loop(env, save_dir, exp_name, trajectories_needed)

    # 停止监听器
    listener.stop()


"""
手动控制的轨迹收集

使用键盘监听器（'a'开始，'b'停止，'d'删除）

每条轨迹单独保存为一个 .pkl 文件
文件命名：{exp_name}_traj_{编号}.pkl
保存路径：./classifier_data/{日期}/
"""
