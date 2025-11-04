import os
import numpy as np
import sys
from rl.envs_store import *


class TrainConfig:

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 256

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5
    image_keys = [
        "wrist",
        "rgb",
        # "scene",
    ]
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]
    checkpoint_period = 1000
    cta_ratio = 2
    random_steps = 0
    # discount = 0.99
    discount = 0.98
    buffer_period = 1000

    def get_environment(self, fake_env=False, train=True):
        proprio_keys = ["tcp_pose", "gripper_pose"]

        env = Fake_UR_Platform_Env()
        env = RelativeFrame(env, include_relative_pose=False)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
