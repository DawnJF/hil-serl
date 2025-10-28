import os
import time
import jax
import numpy as np
import jax.numpy as jnp
import sys
import pickle as pkl
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig, FakeFrankaEnv
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
from PIL import Image

sys.path.append(os.getcwd())
sys.path.append("/home/facelesswei/code/hil-serl")
sys.path.append("/home/facelesswei/code/hil-serl/examples")
from examples.experiments.usb_pickup_insertion.config import UREnvConfig
from utils.tools import print_dict_structure

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import USBEnv, GripperPenaltyWrapper
from experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map


def test_Env():

    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]

    # env = FakeFrankaEnv(config=EnvConfig())
    env = UR_Platform_Env(config=UREnvConfig())
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=-0.02)

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

    # image_w = Image.fromarray(obs["images"]["wrist"].squeeze())
    # image_w.save("/home/facelesswei/code/hil-serl/image_w.png")
    # image_r = Image.fromarray(obs["images"]["rgb"].squeeze())
    # image_r.save("/home/facelesswei/code/hil-serl/image_r.png")

    print("==== step ====")
    for _ in range(11110):
        start_time = time.perf_counter()
        # action = np.array([0.0000000000000001,0.0,0.0,0.0,0.0,0.0,0.0])
        # action = np.array([0.63226587,  0.9993608,  -0.08116492 , 0.52754897,  0.27143225, -0.7813476,0.6737941])
        # action = np.array([-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0])
        # action = np.array([-0.35847732,-0.41141108, -0.04426178,  0.9987256, -0.91949886,  0.13108964,0.08994653])
        action = env.action_space.sample()
        action[-1] = -1

        obs, reward, done, truncated, info = env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        print(action)
        dt_s = time.perf_counter() - start_time
        print(f"dt_s: {dt_s}")

        # print(obs.keys())
        # print_dict_structure(obs)


"""
[-0.44214267  0.89261407 -0.70602995  0.34876108  0.20455763 -0.4028668
  0.686052  ]
dt_s: 2.0420524930000283
[ 0.63226587  0.9993608  -0.08116492  0.52754897  0.27143225 -0.7813476
  0.6737941 ]
"""


def save_np_as_image(
    array, save_path="examples/experiments/usb_pickup_insertion/debug5"
):
    import os
    import time
    import numpy as np
    from PIL import Image

    """
    将numpy数组保存为图片文件，使用随机时间戳作为文件名

    Args:
        np_array: numpy数组，表示图像
        save_path: 保存路径（目录）

    Returns:
        str: 保存的完整文件路径
    """
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    if isinstance(array, np.ndarray):
        np_array = array

    np_array = np_array.squeeze()
    # 确保数据类型正确
    if np_array.dtype != np.uint8:
        # 如果是浮点数，假设范围在[0,1]或[0,255]
        if np_array.max() <= 1.0:
            np_array = (np_array * 255).astype(np.uint8)
        else:
            np_array = np_array.astype(np.uint8)

    # 使用PIL保存图片
    pil_image = Image.fromarray(np_array)

    # 生成基于时间戳格式化的文件名

    filename = f"image_{time.strftime('%Y%m%d_%H%M%S')}_{pil_image.size}.png"
    full_path = os.path.join(save_path, filename)
    pil_image.save(full_path)

    return full_path


def test_dataset():
    path = "/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-09/usb_pickup_insertion_1_10-39-07.pkl"

    with open(path, "rb") as f:
        transitions = pkl.load(f)
        print(transitions[0].keys())

        for i in range(200):
            print("==== step ", i, " ====")
            print(f"reward: {transitions[i]['rewards']}")
            print(f"done: {transitions[i]['dones']}")
            print(f"action: {transitions[i]['actions']}")
            print(f"masks: {transitions[i]['masks']}")


def replay_dataset():
    path = "datasets/trajectories/2025-10-27/traj_19-22-40_9.pkl"

    proprio_keys = ["tcp_pose", "gripper_pose"]
    env = UR_Platform_Env(config=UREnvConfig())
    # env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=-0.02)

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


if __name__ == "__main__":

    # test_Env()
    # test_dataset()
    replay_dataset()
