import time
from typing import Callable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from gymnasium import Env, spaces

from reward_model.train import RewardModelInferencer


class RewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, model_path, multi=False):
        super().__init__(env)

        self.multi = multi  # 先设置 multi 属性

        self.reward_classifier_func = RewardModelInferencer(model_path=model_path)

    def compute_reward(self, obs):
        if self.multi:
            return self.reward_classifier_func.predict(
                [obs["images"]["rgb"], obs["images"]["wrist"]]
            )
        else:
            return self.reward_classifier_func.predict([obs["images"]["rgb"]])

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info["succeed"] = bool(rew)

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.getcwd())
    sys.path.append("/home/facelesswei/code/hil-serl")
    sys.path.append("/home/facelesswei/code/hil-serl/examples")
    from examples.experiments.usb_pickup_insertion.config import UREnvConfig
    from franka_env.envs.wrappers import SpacemouseIntervention
    from examples.experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env

    env = UR_Platform_Env(fake_env=fake_env, config=UREnvConfig())
    env = HumanRewardEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env, include_relative_pose=False)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    # 包装成带奖励分类器的环境
    env = RewardClassifierWrapper(
        env=env,
        multi=True,
        model_path="/home/facelesswei/code/debug_UR_Robot_Arm_Show/reward_model/checkpoint_plug_multi/checkpoint-1.pth",
    )
    for _ in range(10000):
        action = env.action_space.sample()
        action = np.zeros((7,))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward: {reward}")

        if reward > 0:
            print("succeed!")
            time.sleep(1)
            env.reset()
