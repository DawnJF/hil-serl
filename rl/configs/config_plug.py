import os
import numpy as np
import sys
from rl.envs_store import *


class UREnvConfig:
    REALSENSE_CAMERAS = {
        # "wrist": {"dim": (1280, 720)},
        "rgb": {"dim": (1280, 720)},
        "scene": {"dim": (1280, 720)},
    }
    IMAGE_CROP = {
        # "wrist": lambda img: img[60:350, 50:590],
        "rgb": lambda img: img[250:510, 320:560],
        "scene": lambda img: img[175:480, 0:640],
    }
    # TARGET_POSE = np.array(
    #     [0.553, 0.1769683108549487, 0.25097833796596336, np.pi, 0, -np.pi / 2]
    # )
    # reset_xyz = np.array([-0.35, -0.5, 0.15])
    #  [-0.42, -0.42, 0.181]
    # [-0.4961725175380707, -0.27718037366867065, 0.18155656599998474]
    reset_xyz = np.array(
        [-0.42, -0.42, 0.18]
        # [-0.496, -0.277, 0.181]
    )
    reset_euler = np.array([np.pi, 0, np.pi * 3 / 4])
    # For plug
    reset_quat = np.array(
        [
            0.9824031954007385,
            0.1864148915326869,
            0.010644197470452267,
            0.004488980004955199,
        ]
    )
    RESET_POSE = np.array([*reset_xyz, *reset_quat])
    ACTION_SCALE = np.array([0.007, 0.02, 1])  # xyz, euler, gripper
    GRIPPER_OPEN_POSE = 120
    GRIPPER_CLOSE_POSE = 190
    GRIPPER_SPEED = 30
    GRIPPER_FORCE = 180
    RANDOM_RESET = True
    # RANDOM_RESET = False

    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    # [-0.5, -0.2, 0.25]
    # [-0.6, -0.6, 0.055]
    ABS_POSE_LIMIT_HIGH = np.concatenate(
        [np.array([-0.35, -0.15, 0.20]), reset_euler + np.array([0.1, 0.1, 0.3])]
    )
    ABS_POSE_LIMIT_LOW = np.concatenate(
        [np.array([-0.65, -0.55, 0.075]), reset_euler - np.array([0.1, 0.1, 0.3])]
    )
    MAX_EPISODE_LENGTH = 250


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
        # "wrist",
        "rgb",
        "scene",
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
        env_config = UREnvConfig()
        if not train:
            env_config.MAX_EPISODE_LENGTH = 1000
        env = UR_Platform_Env(fake_env=fake_env, config=env_config)
        # env = HumanControlTargetEnv(env, "1")
        env = HumanRewardEnv(env)
        env = SpacemouseIntervention(env)
        env = RelativeFrame(env, include_relative_pose=False)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = GripperPenaltyWrapper(env, penalty=-0.04)
        return env
