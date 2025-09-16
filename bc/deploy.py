import os
import time
import jax
import numpy as np
import jax.numpy as jnp
import sys
import pickle as pkl
import glob
import torch
from torchvision import transforms
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
from examples.experiments.usb_pickup_insertion.config import EnvConfig, UREnvConfig
from utils.tools import print_dict_structure

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import USBEnv, GripperPenaltyWrapper
from experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env
from bc.net import BCActor
from bc.train import ActorWrapper, Args as BCArgs, load_checkpoint
from utils.image_augmentations import get_eval_transform as get_eval_tf

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map


def test_Env():

    proprio_keys = ["tcp_pose", "gripper_pose"]
    config = UREnvConfig()
    config.MAX_EPISODE_LENGTH = 1000

    env = UR_Platform_Env(config=config)
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

    ckpt_path = (
        "/home/facelesswei/code/bc_outputs/bc_20250915_120346/checkpoint-100.pth"
    )
    model = ActorWrapper(ckpt_path)

    obs, info = env.reset()
    print(obs.keys())
    print_dict_structure(obs)

    print("==== step ====")
    for _ in range(11110):
        start_time = time.perf_counter()

        action = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            obs, info = env.reset()

        # print(obs.keys())
        # print_dict_structure(obs)


if __name__ == "__main__":
    test_Env()
