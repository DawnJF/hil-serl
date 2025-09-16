# 需要把环境中reset的time.sleep()注释一下   推理脚本不使用reward
# /home/facelesswei/code/hil-serl/examples/experiments/usb_pickup_insertion/config.py  MAX_EPISODE_LENGTH = 200改为10000000

import jax
import time

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree.leaves
import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import atexit

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", "usb_pickup_insertion", "Name of experiment corresponding to folder."
)
flags.DEFINE_integer("trajectories_needed", 500, "Number of trajectories to collect.")

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


def main(_):
    global collecting, start_collect, stop_collect, delete_last
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 注册退出时停止监听器
    atexit.register(lambda: listener.stop())

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    trajectories_needed = FLAGS.trajectories_needed

    trajectory = []  # 当前一条轨迹

    # 建立以日期为名的输出目录
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join("./classifier_data", date_str)
    os.makedirs(save_dir, exist_ok=True)

    # 获取已有文件编号，决定下次编号
    existing_files = [
        f
        for f in os.listdir(save_dir)
        if f.startswith(FLAGS.exp_name) and f.endswith(".pkl")
    ]
    if existing_files:
        existing_nums = []
        for f in existing_files:
            try:
                num = int(f.split("_traj_")[1].split(".pkl")[0])
                existing_nums.append(num)
            except:
                pass
        start_idx = max(existing_nums) + 1 if existing_nums else 1
    else:
        start_idx = 1

    saved_count = start_idx - 1
    print(f"继续编号，从 {start_idx} 开始保存")

    # 进度条调整为剩余数量
    pbar = tqdm(total=trajectories_needed, initial=saved_count)

    while saved_count < trajectories_needed:
        # 按 d 删除上一次轨迹
        if delete_last:
            if saved_count > 0:
                prev_file = os.path.join(
                    save_dir, f"{FLAGS.exp_name}_traj_{saved_count}.pkl"
                )
                if os.path.exists(prev_file):
                    os.remove(prev_file)
                    print(f"Deleted previous trajectory: {prev_file}")
                    saved_count -= 1
                    pbar.update(-1)
            else:
                print("No previous trajectory to delete")
            delete_last = False

        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        if (
            not done
            and not rew
            and actions[0] == 0
            and actions[1] == 0
            and actions[2] == 0
            and actions[3] == 0
        ):
            continue

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs

        # 按 a → 开始采集
        if start_collect:
            if not collecting:
                trajectory = []
                collecting = True
                print("Start collecting new trajectory")
            trajectory.append(transition)
            start_collect = False

        # 正在采集 → 每步都保存
        elif collecting:
            trajectory.append(transition)

        # 按 b → 停止采集并保存
        if stop_collect and collecting and len(trajectory) > 0:
            file_name = os.path.join(
                save_dir, f"{FLAGS.exp_name}_traj_{saved_count+1}.pkl"
            )
            with open(file_name, "wb") as f:
                pkl.dump(trajectory, f)
            print(f"=====saved trajectory {saved_count+1} to {file_name}")

            # 重置状态
            trajectory = []
            collecting = False
            stop_collect = False
            saved_count += 1
            pbar.update(1)

            # 确保环境重置
            try:
                obs, _ = env.reset()
                print("环境重置成功")
            except Exception as e:
                print(f"环境重置失败: {e}")
                env = config.get_environment(
                    fake_env=False, save_video=False, classifier=False
                )
                obs, _ = env.reset()

            print("====================收集了一条数据")

        # 如果环境结束 → reset
        if done or truncated:
            obs, _ = env.reset()
            print("环境自然结束，已重置")

    print(f"Finished! Saved {saved_count} trajectories in {save_dir}")
    listener.stop()


if __name__ == "__main__":
    app.run(main)
